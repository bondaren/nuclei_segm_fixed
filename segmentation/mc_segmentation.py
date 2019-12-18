import argparse
import os
import h5py
import nifty
import nifty.graph.rag as nrag

from elf.segmentation.features import compute_rag
from elf.segmentation.multicut import multicut_kernighan_lin, transform_probabilities_to_costs
from elf.segmentation.watershed import distance_transform_watershed


def segment_volume(inp, threshold=0.3, sigma=2.0, beta=0.8, ws=None):
    if ws is None:
        ws, _ = distance_transform_watershed(inp, threshold, sigma, min_size=10)

    rag = compute_rag(ws, 1)
    features = nrag.accumulateEdgeMeanAndLength(rag, inp, numberOfThreads=1)
    probs = features[:, 0]  # mean edge prob
    edge_sizes = features[:, 1]
    costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes, beta=beta)
    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(rag.uvIds())

    print('Solving multicut')
    node_labels = multicut_kernighan_lin(graph, costs)

    return nifty.tools.take(node_labels, ws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MC seg')
    parser.add_argument('--pmaps', type=str, required=True, help='path to the network predictions')
    parser.add_arguemnt('--channel', type=int, required=True, help='Boundary pmaps channel')
    args = parser.parse_args()

    in_file = args.pmaps
    c = args.channel
    out_file = os.path.splitext(in_file)[0] + '_mc.h5'

    with h5py.File(in_file, 'r') as f:
        print(f'Extracting pmaps from: {in_file}')
        pmaps = f['predictions'][c]

    print(f'Running MC...')
    mc = segment_volume(pmaps)
    mc = mc.astype('uint16')
    print(f'Saving results to: {out_file}')

    with h5py.File(out_file, 'w') as f:
        output_dataset = 'segmentation'
        if output_dataset in f:
            del f[output_dataset]
        f.create_dataset(output_dataset, data=mc, compression='gzip')
