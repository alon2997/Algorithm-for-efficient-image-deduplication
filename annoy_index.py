from annoy import AnnoyIndex

def build_annoy_index(data, n_trees):
    assert len(data.shape) == 2
    n_vecs, vec_length = data.shape
    index = AnnoyIndex(vec_length, 'angular')
    for i in range(n_vecs):
        index.add_item(i, data[i])
    index.build(n_trees)
    return index
