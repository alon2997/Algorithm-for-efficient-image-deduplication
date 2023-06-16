import numpy as np
from collections import defaultdict
from PIL import Image
import os
from annoy_index import *
import networkx as nx
import gradio as gr 
from aug_pipeline import *
from collections import defaultdict
import matplotlib.pyplot as plt
import copy 

class Dataset():
    def __init__(self):
        self.data = None
        self.dup_components = None
        self.index = None
        self.img_to_dup = defaultdict(list)
        self.ground_truth = None
        self.embeddings = None
        self.duplicate_graph = nx.Graph()
        self.original_imgs = [] # array of original images

    def remove_idxs_and_get_data(self, idxs):
        if type(self.original_imgs) == np.ndarray:
            mask = np.ones(len(self.data), dtype = bool)
            for idx in idxs:
                mask[idx] = False
            return np.copy(self.original_imgs[mask])
        elif type(self.original_imgs) == list:
            idxs = set(idxs)
            ret = []
            for i in range(len(self.original_imgs)):
                if i not in idxs:
                    ret.append(copy.deepcopy(self.original_imgs[i]))
            return ret
        else:
            print(f"original_imgs are of unrecognized type")
    
    def remove_all_dups_and_get_data(self):
        idxs_to_remove = []
        for cluster in self.dup_components:
            idxs_to_remove.extend(cluster[1:])
        return self.remove_idxs_and_get_data(idxs_to_remove)
       
    def get_all_dup_idxs(self):
        idxs_to_remove = []
        for cluster in self.dup_components:
            idxs_to_remove.extend(cluster[1:])
        return idxs_to_remove
    def read_data_from_array(self, array):
        """
        To conserve space and support large data sets, only resized images
        are kept, slicing original data is up to the user
        """
        print("Copying and resizing data to appropriate size, after extracting indeces to drop, they should be removed from original data")
        self.data = np.copy(array)
        channels = array[0].shape[-1]
        for i in range(self.data.shape[0]):
            if self.data[i].shape[:2] != (32, 32) or self.data[i].shape[-1] != 3:
                self.data[i] = self.data[i].astype(np.uint8)
                self.data[i] = Image.fromarray(self.data[i])
                self.data[i] = self.data[i].resize([32, 32]) # size needs to be (width, height)
                if channels == 1:
                    self.data[i] = self.data[i].convert("RGB") # 1 channel
                elif channels == 3:
                    self.data[i] = self.data[i].convert("RGB") # 3 channels
                self.data[i] = np.array(self.data[i])
        if np.max(self.data) > 1:
            self.data = self.data / 255.
        self.ground_truth = {i : [] for i in range(len(self.data))}

    def read_data_from_dir(self, path):
        """
        Read images from directory and load them into memory 
        """
        self.original_imgs = []
        for file in os.listdir(path): # get a list of all the files in the directory specified by path
            if file.endswith(".jpeg") or file.endswith(".jpg"): # check if the file is a jpg file (can also add file.endswith(".png"))
                img = Image.open(os.path.join(path, file)) # opens the image with PIL library
                img = np.array(img) # convert to numpy array
                self.original_imgs.append(img) # append to the data
        self.resize_data((32,32),3) # Verify all images are appropriate size and resize if not
        self.ground_truth = {i : [] for i in range(len(self.original_imgs))}

    def prepare_duplicates(self, model, thresh):
        if not self.embeddings: self.get_data_embeddings(model)
        self.build_embedding_index()
        self.find_duplicates(thresh)

    def get_data_embeddings(self, model):
        """
        Assign self.embeddings to numpy array of embeddings returned
        from the model (need to make sure model is loaded and avaliable)
        """
        if not model: 
            raise ValueError("model is not loaded or not available")
        self.embeddings = model.predict(self.data, batch_size = 32)
    
    def build_embedding_index(self):
        """
        Build annoy index of embeddings
        """
        self.index = build_annoy_index(self.embeddings, n_trees = 10) 

    def find_duplicates(self,thresh):
        self.duplicate_graph = nx.Graph()
        for img_idx in range(len(self.data)):
            i = 1
            while i < 10:
                neis, dists = self.index.get_nns_by_item(img_idx, 10*i, search_k=-1, include_distances=True)
                if dists[-1] > thresh:
                    break
                else:
                    i += 1
            # Remove self (the nearest neighbor of each node is always itself with a distance of 0)
            neis, dists = neis[1:], dists[1:]
            # Find all distances below threshold 
            for cutoff_idx in range(len(dists)):
                if dists[cutoff_idx] > thresh:
                    break
            neis, dists = neis[:cutoff_idx], dists[:cutoff_idx]
            # Add edges to graph
            while neis:
                self.duplicate_graph.add_edge(img_idx, neis.pop())
        # Return connected components
        self.dup_components = [list(comp) for comp in nx.connected_components(self.duplicate_graph)] 
        
    def resize_data(self,size,channels): 
        """
        Verify all images are appropriate size and resize if not
        """
        # keep track of the original resolution and image of each image before resizing 
        # and prints a message when it changes resolution of an image. the function saves the original images and resolutions in original_imgs and original_resolution.
        # this information can be used later to restore the original resolution of the images if needed.
        # the original image is saved as a copy before it is resized so that we have a copy of the original image with the original resolution that can be used later
        # to restore the original resolution after removing the duplicate images.
        # without the copy the original image would be modified when resizing it, so we wouldn't be able to restore it to the original resolution.
        new_data = []
        for i in range(len(self.original_imgs)):
            original_resolution = self.original_imgs[i].shape[:2]
            img = self.original_imgs[i].copy()
            if img.shape[:2] != size or img.shape[-1] != channels:
                # Create a copy of the image before resizing it
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img = img.resize(size[::-1]) # size needs to be (width, height)
                if channels == 1:
                    img = img.convert("L") # 1 channel
                elif channels == 3:
                    img = img.convert("RGB") # 3 channels
                new_data.append(np.array(img))
                # save the original img, resolution and index
            else:
                new_data.append(img)
        self.data = np.array(new_data)
        if np.max(self.data) > 1:
            self.data = self.data / 255.
    
    def plot_clusters(self):
        for cluster in self.dup_components:
            n = len(cluster)
            imgs = list(self.data[cluster])
            n_cols = max(4, 2)
            n_rows = max((n // 4) + 1 if n % n_cols else n // 4, 2)
            f, axarr = plt.subplots(n_rows, n_cols)
            i = 0
            for row in range(n_rows):
                for col in range(n_cols):
                    axarr[row, col].set_xticks([])
                    axarr[row, col].set_yticks([])
                    if i == len(imgs):
                        continue
                    axarr[row, col].set_title(str(cluster[i]))
                    axarr[row, col].imshow(imgs[i])
                    i += 1
            plt.show()
                

    def start_viz(self):
        displayed_imgs = []
        buff = np.array([np.ones((32, 32, 3)) for _ in range(12)])
        def get_closest_images(idx):
            nonlocal displayed_imgs
            try :
                cluster = list(self.dup_components[int(idx)])
            except:
                return buff
            displayed_imgs = list(map(str, cluster))
            return np.append(self.data[list(cluster)], buff, axis = 0)

        def get_idxs(idx):
            nonlocal displayed_imgs
            return displayed_imgs[int(idx)]

        def update_checkbox(idx):
            nonlocal displayed_imgs
            return gr.CheckboxGroup.update(choices = displayed_imgs)

        def update_label(idx, curr_labels, n):
            nonlocal displayed_imgs
            if n >= len(displayed_imgs):
                return gr.Image.update(label = '~')
            return gr.Image.update(label = displayed_imgs[int(n)])

        def display_clusters():
            ret = []
            for i, comp in enumerate(self.dup_components):
                ret.append(f"#{i} {len(comp)}, {comp}\n")
            return ''.join(ret)

        def regen_graph(thresh):
            self.find_duplicates(thresh)

        with gr.Blocks() as demo:
            with gr.Tab('Duplicates'):
                selected = gr.Textbox("", label = 'Indeces to remove')
                curr_imgs_check = gr.CheckboxGroup(choices = [], interactive = True, label = "Select indeces to remove")
                gr.Markdown("Nearest neighbors!")
                inp = gr.Slider(0, 200, step = 1, label = "Cluster Number")
                new_label = 5
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            output_0 = gr.Image()
                            output_1 = gr.Image()
                        with gr.Row():
                            output_2 = gr.Image()
                            output_3 = gr.Image()
                    with gr.Column():
                        with gr.Row():
                            output_4 = gr.Image()
                            output_5 = gr.Image()
                        with gr.Row():
                            output_6 = gr.Image()
                            output_7 = gr.Image()
                    with gr.Column():
                        with gr.Row():
                            output_8 = gr.Image()
                            output_9 = gr.Image()
                        with gr.Row():
                            output_10 = gr.Image()
                            output_11 = gr.Image()
                
                def update_selected(checks):
                    return ", ".join(checks)
                
                dup_button = gr.Button("Show duplicates!")
                graph_button = gr.Button("Regenerate graph with new threshold")
                threshold = gr.Slider(0, 2, step = 0.05, label = "Distance threshold")
                output_0.change(fn = update_checkbox, inputs = inp, outputs = curr_imgs_check)
                curr_imgs_check.change(fn = update_selected, inputs = curr_imgs_check, outputs = selected)
                dup_button.click(fn = get_closest_images, inputs = inp, outputs = [output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9, output_10, output_11])
                inp.change(fn = get_closest_images, inputs = inp, outputs = [output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9, output_10, output_11])
                graph_button.click(fn = regen_graph, inputs = threshold, outputs = None)
                output_0.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(0, visible = False)], outputs = output_0)
                output_1.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(1, visible = False)], outputs = output_1)
                output_2.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(2, visible = False)], outputs = output_2)
                output_3.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(3, visible = False)], outputs = output_3)
                output_4.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(4, visible = False)], outputs = output_4)
                output_5.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(5, visible = False)], outputs = output_5)
                output_6.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(6, visible = False)], outputs = output_6)
                output_7.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(7, visible = False)], outputs = output_7)
                output_8.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(8, visible = False)], outputs = output_8)
                output_9.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(9, visible = False)], outputs = output_9)
                output_10.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(10, visible = False)], outputs = output_10)
                output_11.change(fn = update_label, inputs = [inp, curr_imgs_check, gr.Number(11, visible = False)], outputs = output_11)

            with gr.Tab("Clusters Found"):
                clusters_btn = gr.Button("Show clusters")
                clusters_out = gr.Text()
                clusters_btn.click(fn = display_clusters, inputs = None, outputs = clusters_out)

        demo.launch()

    def inject_duplicates(self, n_duplicates):
        """
        Inject augmented duplicates into the dataset and return their idxs
        """    
        duplicate_idxs = np.random.choice(len(self.data), n_duplicates)
        for idx in duplicate_idxs:
            identical = np.random.randint(0, 2) # 0 or 1
            new_img = self.data[idx].copy()
            if not identical:
                new_img = aug_pipeline(new_img)                    
            self.data = np.append(self.data, np.expand_dims(new_img, axis = 0), axis = 0)
            self.img_to_dup[idx].append((self.data.shape[0] - 1, identical))
            self.ground_truth[idx].append(self.data.shape[0] - 1)
            self.same_pairs = {(k) : temp for k, dups in self.img_to_dup.items() if (temp := list(filter(None, [v[0] if v[1] == 1 else None for v in dups])))}
            self.augmented_pairs = {(k) : temp for k, dups in self.img_to_dup.items() if (temp := list(filter(None, [v[0] if v[1] == 0 else None for v in dups])))}
        return duplicate_idxs 

    def benchmark(self, ground_truth):
        """
        Recieves a ground truth dictionary mapping original : list of copies
        and a networkx graph representing our found clusters
        goes over each key value pair in the dict and checks for correctness against
        our graph
        """
        tp_set, fp_set, tn_set, fn_set = set(), set(), set(), set()
        sort_2 = lambda x, y: (x, y) if x < y else (y, x)
        for k, v in ground_truth.items():
            actual_dups = v.copy()
            # Ground truth says there are no dups
            if not actual_dups:
                try:
                    # Try and retrive duplicate cluster
                    matching_dup_cluster = nx.node_connected_component(self.duplicate_graph, k)
                    # Duplicate cluster retrieved -> False positives
                    for dup in matching_dup_cluster:
                        if dup != k:
                            pair = sort_2(k, dup)
                            fp_set.add(pair)

                # Lonely sample was classified correctly
                except:
                    tn_set.add(k)

            # Sample has actual duplicates
            else:
                # Try and get matching cluster, there are three options:
                # Could not find cluster: This is bad and all pairs will be classified as false negatives
                # Found cluster, for each dup in cluster either:
                #   It's found in actual_dups -> This is a true positive
                #   It's not found in actual_dups -> This is a false positive
                try:
                    matching_dup_cluster = nx.node_connected_component(self.duplicate_graph, k)
                    for dup in matching_dup_cluster:
                        pair = sort_2(k, dup)
                        if dup == k:
                            continue
                        elif dup in actual_dups:
                            tp_set.add(pair)
                            actual_dups.remove(dup)
                        # Dup not in actual_dups
                        else:
                            fp_set.add(pair)
                # Found no matching cluster, each k, v pair is a false negative
                except:
                    for dup in actual_dups:
                        fn_set.add(sort_2(k, dup))

            # Samples remained in actual dups -> These are false negatives
            if actual_dups:
                for remainder in actual_dups:
                    pair = sort_2(k, remainder)
                    fn_set.add(pair)

        print(f"{len(tp_set)} | True positives are: {tp_set}")
        print(f"{len(fp_set)} | False positives are: {fp_set}")
        print(f"true negatives: {len(tn_set)}")
        print(f"{len(fn_set)} | False negatives are: {fn_set}")
        return tp_set, fp_set, tn_set, fn_set
