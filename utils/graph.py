def pop_paris(node_data, node_mask) : 
    for i, (v1, v2) in enumerate(node_data) :
        # 삭제할 노드가 있으면
        if (v1 not in node_mask) or (v2 not in node_mask) : 
            del node_data[i]
            pop_paris(node_data, node_mask)
            break
    return 

def sort_paris(node_data, node_mask) : 
    indexTo = [i for i in range(1, len(node_mask) + 1)]
    for i, (v1, v2) in enumerate(node_data) :
        node_data[i] = (indexTo[node_mask.index(v1)], indexTo[node_mask.index(v2)])
    return node_data

def plus_paris(node_data, values) : 
    for i, (v1, v2) in enumerate(node_data) : 
        node_data[i] = (v1 + values, v2 + values)
    return node_data
