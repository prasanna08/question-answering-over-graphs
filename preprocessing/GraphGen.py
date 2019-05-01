from collections import defaultdict

def generate_graph_edges(dependencies, idx=0):
    deps = []
    for dep in dependencies:
        if dep['governorGloss'] != 'ROOT':
            deps.append(((dep['governorGloss'], idx, dep['governor']), (dep['dependentGloss'], idx, dep['dependent']), dep['dep']))
        else:
             deps.append(((dep['dependentGloss'], idx, dep['dependent']), (dep['dependentGloss'], idx, dep['dependent']), dep['dep']))
    return deps


def get_edge_list(qdata, depparsed, annotator, depth=5, n_neighbors=100, dep_type='basicDependencies'):
    graph_edges = defaultdict(list)

    # Add dependency graphs for each of the nearest neighbor sentences.
    for neighbor in qdata['neighbors'][:n_neighbors]:
        for idx, dep in enumerate(depparsed[str(neighbor)]):
            graph_edges[neighbor].extend(generate_graph_edges(dep, idx))
    
    # Anchor tokens are initial tokens extracted from question and choices.
    # Any node which is depth-hop away from these anchors tokens is added in the graph.
    anchors = set(qdata['tagged_tokens'])
    new_edges = defaultdict(list)
    for _ in range(depth):
        for neighbor in graph_edges:
            for edge in graph_edges[neighbor]:
                if edge[0][0] in anchors or edge[1][0] in anchors:
                    new_edges[neighbor].append(edge)
        new_anchors = set()
        for edges in new_edges.values():
            for e1, e2, _ in edges:
                new_anchors.add(e1[0])
                new_anchors.add(e2[0])
        anchors = new_anchors
    
    # Add depedency graphs for question and chocies as well.
    dep = annotator(qdata['question']['stem'])
    for idx, sentence in enumerate(dep['sentences']):
            new_edges['question'].extend(generate_graph_edges(sentence[dep_type], idx))
    for choice in qdata['question']['choices']:
        dep = annotator(choice['text'])
        for idx, sentence in enumerate(dep['sentences']):
            new_edges['choice:%s' % choice['label']].extend(generate_graph_edges(sentence[dep_type], idx))
    return new_edges
    
def prune_graph(qdata, tags_to_remove, stop_words):
    for neighbor in qdata['edges']:
        nodes_to_neighbors = defaultdict(set)
        if type(neighbor) == str and (neighbor.startswith('choice') or neighbor.startswith('question')):
            new_edges = [(e1, e2) for e1, e2, _ in qdata['edges'][neighbor]]
            qdata['edges'][neighbor] = new_edges
            continue
        # Create adjacency list representation of edges for each neighbor ndoe.
        for e1, e2, r in qdata['edges'][neighbor]:
            if r not in tags_to_remove:
                nodes_to_neighbors[e1].add(e2)
        
        # Remove stop_word dependency by directly joining two words if they are connected by stop words.
        for e1 in nodes_to_neighbors:
            new_neighbors = nodes_to_neighbors[e1]
            for e2 in nodes_to_neighbors[e1]:
                if e2[0] in stop_words and e2 in nodes_to_neighbors:
                    new_neighbors = new_neighbors.union(nodes_to_neighbors[e2])
            # Remove stop words from neighbors of current word.
            new_new_neighbors = set()
            for e2 in new_neighbors:
                if e2[0] in stop_words:
                    continue
                new_new_neighbors.add(e2)
            nodes_to_neighbors[e1] = new_new_neighbors
        
        # Remove any stop words from overall nodes.
        remove_nodes = []
        for e1 in nodes_to_neighbors:
            if e1[0] in stop_words:
                remove_nodes.append(e1)
        for e1 in remove_nodes:
            del nodes_to_neighbors[e1]
        # Generate new edges among new set of nodes.
        new_edges = set()
        for e1 in nodes_to_neighbors:
            for e2 in nodes_to_neighbors[e1]:
                if e1[2] != e2[2]:
                    new_edges.add((e1, e2))
        qdata['edges'][neighbor] = list(new_edges)

def get_final_graph_representation(qdata, directed=False):
    nodes_text = set()
    nodes_with_text = defaultdict(set)
    sentential_nodes = set()
    graph = []
    for n in qdata['edges']:
        for e1, e2 in qdata['edges'][n]:
            nodes_text.add(e1[0])
            nodes_text.add(e2[0])
            sentential_nodes.add((n, e1[2]))
            sentential_nodes.add((n, e2[2]))
        sentential_nodes_to_idx = dict((v, i) for i, v in enumerate(sentential_nodes))

        # Find out all the nodes having same token text. They basically have match based edges among them.
        for e1, e2 in qdata['edges'][n]:
            nodes_with_text[e1[0]].add((e1[0], n, e1[2]))
            nodes_with_text[e2[0]].add((e2[0], n, e2[2]))

        # Remove any token that has less than 2 appearances. It means that it is
        # not linked with anyone else in match based edges.
        remove_words = []
        for word in nodes_with_text:
            if len(nodes_with_text[word]) < 2:
                remove_words.append(word)
        for word in remove_words:
            del nodes_with_text[word]

    # Find out doc based neighbors and match based neighbors.
    doc_neighbors = defaultdict(set)
    match_neighbors = defaultdict(set)
    for n in qdata['edges']:
        # All tokens in edges are related by doc based neighbors.
        for e1, e2 in qdata['edges'][n]:
            u = sentential_nodes_to_idx[(n, e1[2])]
            v = sentential_nodes_to_idx[(n, e2[2])]
            if u == v:
                continue
            doc_neighbors[u].add(v)
            if not directed:
                doc_neighbors[v].add(u)

    # All tokens in nodes_with_text with different sentence idx are related by match based edges.
    for word in nodes_with_text:
        for u in nodes_with_text[word]:
            for v in nodes_with_text[word]:
                if u[1] == v[1]:
                    continue
                ui = sentential_nodes_to_idx[u[1:]]
                vi = sentential_nodes_to_idx[v[1:]]
                match_neighbors[ui].add(vi)
                match_neighbors[vi].add(ui)
    return sentential_nodes_to_idx, doc_neighbors, match_neighbors
