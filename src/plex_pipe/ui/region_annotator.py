import napari
from napari.layers import Labels
from magicgui import magicgui
import spatialdata as sd
import numpy as np
import pandas as pd
from spatialdata.models import Labels2DModel, TableModel
import anndata as ad
from pathlib import Path

class RegionAnnotationWidget:
    def __init__(self, viewer: napari.Viewer, sdata: sd.SpatialData, image_name: str, img_shape: tuple):
        self.viewer = viewer
        self.sdata = sdata
        self.image_name = image_name
        self.annotations = {}
        
        # Check if tissue_regions already exist
        existing_regions = 'tissue_regions' in sdata.labels
        
        if existing_regions:
            print("Loading existing tissue regions...")
            # Load existing mask using SpatialData API
            existing_mask = np.array(sd.get_pyramid_levels(sdata.labels['tissue_regions'], n=0)).squeeze()
            
            # Find the max region ID
            self.current_region_id = int(np.max(existing_mask))
            
            # Load existing annotations if they exist
            if 'tissue_regions_table' in sdata.tables:
                region_table = sdata.tables['tissue_regions_table']
                for idx, row in region_table.obs.iterrows():
                    region_id = int(row['region_id'])
                    self.annotations[region_id] = {
                        'region_id': region_id,
                        'tissue_type': row.get('tissue_type', 'Unknown'),
                        'phenotype': row.get('phenotype', ''),
                        'additional_notes': row.get('notes', '')
                    }
                print(f"Loaded {len(self.annotations)} existing annotations")
            
            # Create labels layer with existing data
            self.regions_layer = self.viewer.add_labels(
                existing_mask,
                name='tissue_regions'
            )
        else:
            print("Starting fresh annotations...")
            self.current_region_id = 1
            # Create empty labels layer
            self.regions_layer = self.viewer.add_labels(
                np.zeros(img_shape, dtype=np.int32),
                name='tissue_regions'
            )
        
        # Optionally load existing cell segmentation for reference
        if 'instanseg_cell' in sdata.labels:
            try:
                cell_data = np.array(sd.get_pyramid_levels(sdata.labels['instanseg_cell'], n=0)).squeeze()
                self.viewer.add_labels(
                    cell_data,
                    name='cells',
                    opacity=0.3
                )
            except Exception as e:
                print(f"Could not load cell segmentation: {e}")
        
        self.regions_layer.mode = 'paint'
        self.regions_layer.selected_label = self.current_region_id
        
        self._create_widget()
        
    def _create_widget(self):
        @magicgui(
            call_button="Annotate Current Region",
            tissue_type={'choices': ['Duct', 'Lobule', 'Stroma', 'Other']},
            phenotype={'label': 'Phenotype Notes'},
            additional_notes={'label': 'Additional Notes', 'widget_type': 'TextEdit'}
        )
        def annotation_widget(
            tissue_type: str = 'Duct',
            phenotype: str = '',
            additional_notes: str = ''
        ):
            """Annotate the current region being painted"""
            region_id = self.regions_layer.selected_label
            
            if region_id == 0:
                print("Please select a non-zero label!")
                return
            
            # Check if updating existing annotation
            if region_id in self.annotations:
                print(f"Updating annotation for region {region_id}")
            else:
                print(f"Creating new annotation for region {region_id}")
            
            self.annotations[region_id] = {
                'region_id': region_id,
                'tissue_type': tissue_type,
                'phenotype': phenotype,
                'additional_notes': additional_notes
            }
            
            print(f"Annotated region {region_id}: {tissue_type} - {phenotype}")
        
        @magicgui(call_button="New Region (Next ID)")
        def new_region_widget():
            """Create a new region with next available ID"""
            # Find the next available ID
            mask_data = self.regions_layer.data
            existing_ids = set(np.unique(mask_data))
            existing_ids.update(self.annotations.keys())
            existing_ids.discard(0)  # Remove background
            
            if len(existing_ids) == 0:
                self.current_region_id = 1
            else:
                self.current_region_id = max(existing_ids) + 1
            
            self.regions_layer.selected_label = self.current_region_id
            print(f"Started new region: {self.current_region_id}")
            print(f"Paint your region, then click 'Annotate Current Region'")
        
        @magicgui(
            call_button="Select Region to Work On",
            region_id={'label': 'Region ID'}
        )
        def select_region_widget(region_id: int = 1):
            """Select a specific region ID to paint/edit"""
            self.regions_layer.selected_label = region_id
            self.current_region_id = region_id
            
            if region_id in self.annotations:
                ann = self.annotations[region_id]
                print(f"\nSelected existing region {region_id}:")
                print(f"  Tissue type: {ann['tissue_type']}")
                print(f"  Phenotype: {ann['phenotype']}")
                print(f"  Notes: {ann['additional_notes']}")
                print("\nYou can now:")
                print("  - Paint to add/modify the mask")
                print("  - Use eraser to remove parts")
                print("  - Click 'Annotate Current Region' to update metadata")
            else:
                print(f"\nSelected region {region_id} (new region)")
                print("Paint your region, then click 'Annotate Current Region'")
        
        @magicgui(call_button="List All Regions")
        def list_regions_widget():
            """Show all existing regions"""
            mask_data = self.regions_layer.data
            mask_ids = set(np.unique(mask_data))
            mask_ids.discard(0)
            
            annotated_ids = set(self.annotations.keys())
            all_ids = mask_ids.union(annotated_ids)
            
            if len(all_ids) == 0:
                print("No regions exist yet")
                return
            
            print("\n=== Existing Regions ===")
            for region_id in sorted(all_ids):
                has_mask = region_id in mask_ids
                has_annotation = region_id in annotated_ids
                
                status_parts = []
                if has_mask:
                    status_parts.append("has mask")
                if has_annotation:
                    ann = self.annotations[region_id]
                    status_parts.append(f"{ann['tissue_type']}")
                
                status = ", ".join(status_parts) if status_parts else "no data"
                
                current_marker = " ← CURRENT" if region_id == self.regions_layer.selected_label else ""
                print(f"  Region {region_id}: {status}{current_marker}")
        
        @magicgui(
            call_button="Delete Region",
            region_id={'label': 'Region ID to delete'}
        )
        def delete_region_widget(region_id: int = 1):
            """Delete a region and its annotation"""
            # Remove from mask
            mask_data = self.regions_layer.data
            if region_id in np.unique(mask_data):
                mask_data[mask_data == region_id] = 0
                self.regions_layer.data = mask_data
                print(f"Removed region {region_id} from mask")
            
            # Remove annotation
            if region_id in self.annotations:
                del self.annotations[region_id]
                print(f"Deleted annotation for region {region_id}")
            
            if region_id not in np.unique(mask_data) and region_id not in self.annotations:
                print(f"Region {region_id} fully deleted")
        
        @magicgui(call_button="Save Regions to SpatialData")
        def save_widget():
            """Save region masks and annotations"""
            if len(self.annotations) == 0:
                print("No annotations to save!")
                return
            
            mask_data = self.regions_layer.data.copy()  # Make a copy so we can modify it
            
            # Always set pixel (0,0) to region 0 as background
            mask_data[0, 0] = 0
            
            unique_labels = np.unique(mask_data)
            unique_labels = unique_labels[unique_labels != 0]
            
            if len(unique_labels) == 0:
                print("No regions drawn!")
                return
            
            # Warn about regions without annotations
            unannotated = [int(label_id) for label_id in unique_labels if label_id not in self.annotations]
            if unannotated:
                print(f"⚠️  Warning: Regions {unannotated} have masks but no annotations")
            
            # Preserve existing transformations and attributes
            transformations = None
            existing_chunks = None
            existing_scale_factors = None
            
            # Check if tissue_regions already exists and preserve its attributes
            if 'tissue_regions' in self.sdata.labels:
                existing_labels = self.sdata.labels['tissue_regions']
                try:
                    # Preserve transformations from existing labels
                    if hasattr(existing_labels, 'transform'):
                        transformations = existing_labels.transform
                    # Try to preserve chunk size and scale factors from existing
                    if hasattr(existing_labels, 'chunks'):
                        existing_chunks = existing_labels.chunks
                    # Get scale factors from existing pyramid structure
                    try:
                        # Try to infer from existing data structure
                        existing_scale_factors = [2, 2]  # Default, will try to preserve if possible
                    except:
                        pass
                except Exception as e:
                    print(f"Note: Could not read attributes from existing labels: {e}")
            
            # If no existing transformations, try to get from reference image
            if transformations is None:
                try:
                    ref_element = self.sdata.images[self.image_name]
                    if hasattr(ref_element, 'transform'):
                        transformations = ref_element.transform
                except Exception as e:
                    print(f"Note: Could not copy transformations from reference image: {e}")
            
            # Save labels with proper SpatialData model
            # Get reference shape to determine chunk size (use reasonable defaults)
            ref_shape = mask_data.shape
            if existing_chunks is not None:
                chunks = existing_chunks
            else:
                chunks = (min(1024, ref_shape[0]), min(1024, ref_shape[1]))
            
            # Use existing scale factors if available, otherwise default
            scale_factors = existing_scale_factors if existing_scale_factors is not None else [2, 2]
            
            # Create labels model
            labels_model = Labels2DModel.parse(
                data=mask_data.astype(np.int32),
                dims=("y", "x"),
                scale_factors=scale_factors,
                chunks=chunks,
            )
            # Set transformations if we have them
            if transformations is not None:
                labels_model.transform = transformations
            
            # Update labels (this preserves other sdata attributes)
            self.sdata.labels['tissue_regions'] = labels_model
            
            # Create or update region-level table
            # Check if table already exists and merge data
            existing_table = None
            if 'tissue_regions_table' in self.sdata.tables:
                existing_table = self.sdata.tables['tissue_regions_table']
                print("Found existing table, merging annotations...")
            
            # Build new observation data
            # ALWAYS start with index 0 as Unknown background
            # Then add all painted regions
            
            obs_data = []
            
            # Index 0: Always Unknown background (even if not painted)
            obs_data.append({
                'region_id': 0,
                'area': 0,
                'tissue_type': 'Unknown',
                'phenotype': '',
                'notes': 'Background/Unpainted'
            })
            
            # Now add all painted regions (1, 2, 3, ...)
            for label_id in sorted(unique_labels):
            
                mask = mask_data == label_id
                area = np.sum(mask)
                
                # Check if this region exists in existing table
                existing_row = None
                if existing_table is not None:
                    existing_rows = existing_table.obs[existing_table.obs['region_id'] == label_id]
                    if len(existing_rows) > 0:
                        existing_row = existing_rows.iloc[0]
                
                # Use annotation if available, otherwise check existing table, otherwise defaults to Unknown
                if label_id in self.annotations:
                    ann = self.annotations[label_id]
                elif existing_row is not None:
                    ann = {
                        'tissue_type': existing_row.get('tissue_type', 'Unknown'),
                        'phenotype': existing_row.get('phenotype', ''),
                        'additional_notes': existing_row.get('notes', '')
                    }
                else:
                    # Default to Unknown if not annotated
                    ann = {
                        'tissue_type': 'Unknown',
                        'phenotype': '',
                        'additional_notes': ''
                    }
                
                obs_data.append({
                    'region_id': int(label_id),
                    'area': area,
                    'tissue_type': ann['tissue_type'],
                    'phenotype': ann['phenotype'],
                    'notes': ann['additional_notes']
                })
            
            obs_df = pd.DataFrame(obs_data)
            # Simple approach: just use default integer index (0, 1, 2...)
            # region_id stays as a column
            obs_df = obs_df.reset_index(drop=True)
            
            # Convert categorical columns to categorical dtype for napari-spatialdata
            obs_df['tissue_type'] = pd.Categorical(obs_df['tissue_type'])
            obs_df['phenotype'] = pd.Categorical(obs_df['phenotype'])
            
            # Add the region column that spatialdata expects
            obs_df['region'] = 'tissue_regions'
            
            # Preserve existing uns metadata if available (but NOT spatialdata_attrs)
            uns_metadata = {}
            if existing_table is not None:
                # Get the underlying AnnData object
                existing_adata = existing_table.adata if hasattr(existing_table, 'adata') else None
                if existing_adata is not None and hasattr(existing_adata, 'uns'):
                    # Merge with existing uns metadata, EXCLUDING spatialdata_attrs
                    existing_uns = existing_adata.uns
                    uns_metadata.update({k: v for k, v in existing_uns.items() if k != 'spatialdata_attrs'})
            
            # Create AnnData WITHOUT spatialdata_attrs in uns
            adata = ad.AnnData(
                X=np.zeros((len(obs_df), 0)),
                obs=obs_df,
                uns=uns_metadata
            )
            
            # Parse table model - let TableModel.parse() set spatialdata_attrs
            # region_id is a column, region_key points to 'region' column
            table_model = TableModel.parse(
                adata,
                region='tissue_regions',
                region_key='region',
                instance_key='region_id'
            )
            
            # Update table (this preserves other sdata attributes)
            self.sdata.tables['tissue_regions_table'] = table_model
            
            print(f"✓ Saved {len(unique_labels)} regions to sdata object (in memory)")
            print("✓ Preserved existing attributes and transformations")
            print("Close napari and choose save option to write to disk")
        
        @magicgui(call_button="Show Summary")
        def summary_widget():
            """Show summary"""
            if len(self.annotations) == 0:
                print("No annotations yet!")
                return
            
            print("\n=== Region Annotations Summary ===")
            tissue_counts = {}
            for ann in self.annotations.values():
                tissue = ann['tissue_type']
                tissue_counts[tissue] = tissue_counts.get(tissue, 0) + 1
            
            print("\nCounts by tissue type:")
            for tissue, count in tissue_counts.items():
                print(f"  {tissue}: {count}")
            
            print("\nDetailed annotations:")
            for region_id in sorted(self.annotations.keys()):
                ann = self.annotations[region_id]
                print(f"\nRegion {region_id}:")
                print(f"  Type: {ann['tissue_type']}")
                print(f"  Phenotype: {ann['phenotype']}")
                print(f"  Notes: {ann['additional_notes']}")
        
        # Add widgets to viewer
        self.viewer.window.add_dock_widget(new_region_widget, name='New Region', area='right')
        self.viewer.window.add_dock_widget(select_region_widget, name='Select Region', area='right')
        self.viewer.window.add_dock_widget(list_regions_widget, name='List Regions', area='right')
        self.viewer.window.add_dock_widget(annotation_widget, name='Annotate', area='right')
        self.viewer.window.add_dock_widget(delete_region_widget, name='Delete Region', area='right')
        self.viewer.window.add_dock_widget(save_widget, name='Save', area='right')
        self.viewer.window.add_dock_widget(summary_widget, name='Summary', area='right')

def launch_region_annotation(sdata_path, auto_backup=True):
    """Launch napari for region annotation with all channels
    
    Parameters
    ----------
    sdata_path : str or Path
        Path to the SpatialData object
    auto_backup : bool
        If True, automatically suggests saving to a new path with '_annotated' suffix
    """
    sdata = sd.read_zarr(sdata_path)
    viewer = napari.Viewer()
    
    # Define colormaps for common markers
    colormaps = {
        'DAPI': 'blue',
        'CK818': 'green', 
        'CK14': 'cyan',
        'ECad': 'red',
        'ER': 'yellow',
        'HER2': 'magenta',
        'GATA3': 'green',
        'FOXA1': 'cyan',
        'AR': 'red',
        'AP2A': 'yellow',
        'AP2B': 'magenta',
        'CD45': 'red'
    }
    
    # Load all images
    print("Loading channels:")
    for img_name in sdata.images.keys():
        print(f"  - {img_name}")
        
        # Get the image data using SpatialData API
        img_data = np.array(sd.get_pyramid_levels(sdata.images[img_name], n=0)).squeeze()
        
        # If there's a channel dimension and it's size 1, squeeze it out
        if img_data.ndim == 3 and img_data.shape[0] == 1:
            img_data = img_data[0]
        
        viewer.add_image(
            img_data,
            name=img_name,
            colormap=colormaps.get(img_name, 'gray'),
            blending='additive',
            visible=(img_name == 'DAPI')
        )
    
    # Use first image for shape reference
    ref_image_name = list(sdata.images.keys())[0]
    ref_img_data = np.array(sd.get_pyramid_levels(sdata.images[ref_image_name], n=0)).squeeze()
    if ref_img_data.ndim == 3 and ref_img_data.shape[0] == 1:
        ref_img_data = ref_img_data[0]
    img_shape = ref_img_data.shape  # (y, x)
    
    # Now create the widget with the correct shape
    widget = RegionAnnotationWidget(viewer, sdata, ref_image_name, img_shape)
    
    print("\n=== Napari Region Annotation Tool ===")
    print("\nWorkflow:")
    print("1. Click 'New Region' to start with next available ID")
    print("2. OR click 'Select Region' to choose a specific ID to work on")
    print("3. Paint your region with the brush tool")
    print("4. Click 'Annotate Current Region' and fill in metadata")
    print("5. Click 'List Regions' to see all existing regions")
    print("6. Repeat for all regions")
    print("7. Click 'Save' to save to sdata (in memory)")
    print("8. Close napari to write to disk")
    print("\nTips:")
    print("- You can toggle channel visibility to see tissue structure")
    print("- Use eraser tool to fix mistakes")
    print("- The current region ID is shown in napari's label controls")
    
    viewer.show(block=True)
    
    print("\n=== Napari closed ===")
    
    # Suggest a safe default path
    original_path = Path(sdata_path)
    if auto_backup:
        suggested_path = original_path.parent / f"{original_path.stem}_annotated.zarr"
    else:
        suggested_path = original_path
    
    print(f"\nOriginal path: {original_path}")
    print(f"Suggested save path: {suggested_path}")
    
    save_choice = input("\nSave options:\n  1. Save to suggested path (safe)\n  2. Overwrite original\n  3. Specify custom path\n  4. Don't save\nChoice (1-4): ")
    
    if save_choice == '1':
        sdata.write(suggested_path)
        print(f"✓ Saved to {suggested_path}")
    elif save_choice == '2':
        confirm = input(f"⚠️  Really overwrite {original_path}? Type 'yes' to confirm: ")
        if confirm == 'yes':
            sdata.write(original_path)
            print(f"✓ Saved to {original_path}")
        else:
            print("Save cancelled")
    elif save_choice == '3':
        custom_path = input("Enter path: ")
        custom_path = Path(custom_path)
        if not custom_path.is_absolute():
            custom_path = original_path.parent / custom_path
        sdata.write(custom_path)
        print(f"✓ Saved to {custom_path}")
    else:
        print("Changes not saved to disk")
    
    return sdata

def export_regions_to_spatialdata(sdata, mask_data, annotations, image_name, 
                                  region_labels_name='tissue_regions',
                                  table_name='tissue_regions_table'):
    """
    Export drawn regions and annotations to SpatialData object without overwriting attributes.
    
    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to update
    mask_data : np.ndarray
        The region mask array (2D integer array)
    annotations : dict
        Dictionary mapping region_id to annotation dict with keys:
        'region_id', 'tissue_type', 'phenotype', 'additional_notes'
    image_name : str
        Name of reference image for coordinate system
    region_labels_name : str
        Name for the labels layer
    table_name : str
        Name for the annotations table
        
    Returns
    -------
    SpatialData
        Updated SpatialData object
    """
    unique_labels = np.unique(mask_data)
    unique_labels = unique_labels[unique_labels != 0]
    
    if len(unique_labels) == 0:
        print("No regions to export!")
        return sdata
    
    # Always set pixel (0,0) to region 0 as background
    mask_data = mask_data.copy()
    mask_data[0, 0] = 0
    
    # Preserve existing transformations and attributes for labels
    transformations = None
    existing_chunks = None
    existing_scale_factors = None
    
    if region_labels_name in sdata.labels:
        existing_labels = sdata.labels[region_labels_name]
        try:
            if hasattr(existing_labels, 'transform'):
                transformations = existing_labels.transform
            if hasattr(existing_labels, 'chunks'):
                existing_chunks = existing_labels.chunks
            existing_scale_factors = [2, 2]  # Default
        except Exception as e:
            print(f"Note: Could not read attributes from existing labels: {e}")
    
    if transformations is None:
        try:
            ref_element = sdata.images[image_name]
            if hasattr(ref_element, 'transform'):
                transformations = ref_element.transform
        except Exception:
            pass
    
    # Create labels model
    ref_shape = mask_data.shape
    chunks = existing_chunks if existing_chunks is not None else (min(1024, ref_shape[0]), min(1024, ref_shape[1]))
    scale_factors = existing_scale_factors if existing_scale_factors is not None else [2, 2]
    
    labels_model = Labels2DModel.parse(
        data=mask_data.astype(np.int32),
        dims=("y", "x"),
        scale_factors=scale_factors,
        chunks=chunks,
    )
    if transformations is not None:
        labels_model.transform = transformations
    
    sdata.labels[region_labels_name] = labels_model
    
    # Create or merge table
    existing_table = None
    if table_name in sdata.tables:
        existing_table = sdata.tables[table_name]
    
    # Add a placeholder at position 0 so actual regions start at index 1
    obs_data = []
    
    # Region 0: Background (pixel 0,0)
    obs_data.append({
        'region_id': 0,
        'area': 1,  # Just pixel (0,0)
        'tissue_type': 'Background',
        'phenotype': '',
        'notes': 'Background region at pixel (0,0)'
    })
    
    # Now add all actual painted regions
    for label_id in sorted(unique_labels):
    
        mask = mask_data == label_id
        area = np.sum(mask)
        
        existing_row = None
        if existing_table is not None:
            existing_rows = existing_table.obs[existing_table.obs['region_id'] == label_id]
            if len(existing_rows) > 0:
                existing_row = existing_rows.iloc[0]
        
        if label_id in annotations:
            ann = annotations[label_id]
        elif existing_row is not None:
            ann = {
                'tissue_type': existing_row.get('tissue_type', 'Unknown'),
                'phenotype': existing_row.get('phenotype', ''),
                'additional_notes': existing_row.get('notes', '')
            }
        else:
            ann = {
                'tissue_type': 'Unknown',
                'phenotype': '',
                'additional_notes': ''
            }
        
        obs_data.append({
            'region_id': int(label_id),
            'area': area,
            'tissue_type': ann['tissue_type'],
            'phenotype': ann['phenotype'],
            'notes': ann['additional_notes']
        })
    
    obs_df = pd.DataFrame(obs_data)
    # Simple approach: just use default integer index (0, 1, 2...)
    # region_id stays as a column
    obs_df = obs_df.reset_index(drop=True)
    
    # Convert categorical columns to categorical dtype for napari-spatialdata
    obs_df['tissue_type'] = pd.Categorical(obs_df['tissue_type'])
    obs_df['phenotype'] = pd.Categorical(obs_df['phenotype'])
    
    # Add the region column that spatialdata expects
    obs_df['region'] = region_labels_name
    
    # Preserve existing uns metadata (but NOT spatialdata_attrs)
    uns_metadata = {}
    if existing_table is not None:
        existing_adata = existing_table.adata if hasattr(existing_table, 'adata') else None
        if existing_adata is not None and hasattr(existing_adata, 'uns'):
            existing_uns = existing_adata.uns
            uns_metadata.update({k: v for k, v in existing_uns.items() if k != 'spatialdata_attrs'})
    
    # Create AnnData WITHOUT spatialdata_attrs in uns
    adata = ad.AnnData(
        X=np.zeros((len(obs_df), 0)),
        obs=obs_df,
        uns=uns_metadata
    )
    
    # Parse table model - let TableModel.parse() set spatialdata_attrs
    # region_id is a column, region_key points to 'region' column
    table_model = TableModel.parse(
        adata,
        region=region_labels_name,
        region_key='region',
        instance_key='region_id'
    )
    
    sdata.tables[table_name] = table_model
    
    print(f"✓ Exported {len(unique_labels)} regions to SpatialData")
    return sdata

def link_cells_to_regions(sdata, cell_labels_name='instanseg_cell', 
                         region_labels_name='tissue_regions',
                         cell_table_name='instanseg_table'):
    """
    Assign each cell to a tissue region and add columns to instanseg_table
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with both cell and region segmentations
    cell_labels_name : str
        Name of the cell labels layer
    region_labels_name : str
        Name of the region labels layer
    cell_table_name : str
        Name of the cell table
    """
    print("Linking cells to regions...")
    
    # Get the masks using SpatialData API
    cell_mask = np.array(sd.get_pyramid_levels(sdata.labels[cell_labels_name], n=0)).squeeze()
    region_mask = np.array(sd.get_pyramid_levels(sdata.labels[region_labels_name], n=0)).squeeze()
    
    # Get the cell table
    cell_table = sdata.tables[cell_table_name]
    
    # Get region annotations
    region_table = sdata.tables['tissue_regions_table']
    
    # For each cell, find which region it belongs to
    cell_ids = cell_table.obs['label'].values
    region_assignments = []
    tissue_types = []
    phenotypes = []
    region_notes = []
    
    for cell_id in cell_ids:
        # Get pixels belonging to this cell
        cell_pixels = cell_mask == cell_id
        
        # Find which region(s) overlap
        overlapping_regions = region_mask[cell_pixels]
        overlapping_regions = overlapping_regions[overlapping_regions != 0]
        
        if len(overlapping_regions) == 0:
            # Cell not in any region
            region_id = 0
            tissue_type = 'Unassigned'
            phenotype = ''
            notes = ''
        else:
            # Use majority vote
            region_id = int(np.bincount(overlapping_regions).argmax())
            
            # Get region metadata from column
            region_row = region_table.obs[region_table.obs['region_id'] == region_id]
            if len(region_row) > 0:
                tissue_type = region_row['tissue_type'].values[0]
                phenotype = region_row['phenotype'].values[0]
                notes = region_row['notes'].values[0]
            else:
                tissue_type = 'Unknown'
                phenotype = ''
                notes = ''
        
        region_assignments.append(region_id)
        tissue_types.append(tissue_type)
        phenotypes.append(phenotype)
        region_notes.append(notes)
    
    # Add to cell table
    cell_table.obs['region_id'] = region_assignments
    cell_table.obs['region_tissue_type'] = tissue_types
    cell_table.obs['region_phenotype'] = phenotypes
    cell_table.obs['region_notes'] = region_notes
    
    print(f"Linked {len(cell_ids)} cells to regions")
    print(f"\nDistribution:")
    print(cell_table.obs['region_tissue_type'].value_counts())
    
    return sdata