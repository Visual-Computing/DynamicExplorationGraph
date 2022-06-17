package com.vc.deg.ref.graph;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.function.IntConsumer;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.graph.NodeIds;
import com.vc.deg.graph.NodeView;
import com.vc.deg.graph.WeightedEdgeConsumer;
import com.vc.deg.graph.WeightedEdges;
import com.vc.deg.graph.WeightedUndirectedGraph;




/**
 * This reference implementation uses the label of the data as its identifier for the nodes.
 * Every node contains of a label, data (stored in a MemoryView) and neighbors. The neighbors
 * are just the neighbor label and the edge weight.
 * 
 * 
 * Der Interne Graph mit seinen addNode,deleteNode,addEdge,deleteEdge,serialize Funktionen.
 * Stelle keine Such oder Navigationsmöglichen.
 * 
 * @author Nico Hezel
 *
 */
public class EvenRegularWeightedUndirectedGraph implements Serializable, WeightedUndirectedGraph {

	private static final long serialVersionUID = -9100450642805380353L;

	/**
	 * Label of the node is the identifier
	 */
	protected final Map<Integer, NodeData> nodes;
	
	/**
	 * The feature space knows who many bytes the node data (MemoryView) contains and how to compute the distance between two data objects.
	 */
	protected final FeatureSpace space;
	
	/**
	 * The number of edges per node is fixed and an even number
	 */
	protected final int edgesPerNode;

	public EvenRegularWeightedUndirectedGraph(int edgesPerNode, FeatureSpace space) {
		this.edgesPerNode = edgesPerNode;
		this.nodes = new HashMap<>();
		this.space = space;
	}

	public EvenRegularWeightedUndirectedGraph(int edgesPerNode, int expectedSize, FeatureSpace space) {
		this.edgesPerNode = edgesPerNode;
		this.nodes = new HashMap<>(expectedSize);	
		this.space = space;
	}



	public FeatureSpace getFeatureSpace() {
		return space;
	}

	@Override
	public NodeView getNode(int label) {
		return nodes.getOrDefault(label, null);
	}

	
	@Override
	public boolean addNode(int label, FeatureVector data) {
		if(hasNode(label) == false) {
			nodes.put(label, new NodeData(label, data, edgesPerNode+1));
			return true;
		}
		return false;
	}

	@Override
	public NodeIds getNodeIds() {
		return new SetBasedIds(nodes.keySet());
	}

	@Override
	public boolean hasNode(int id) {
		return nodes.containsKey(id);
	}

	@Override
	public boolean hasEdge(int id1, int id2) {
		final NodeData nodeData = nodes.get(id1);
		if(nodeData == null)
			return false;
		return nodeData.edges.containsKey(id2);
	}

	@Override
	public NodeIds getConnectedNodeIds(int id) {
		return new MapBasedWeighedEdes(id, nodes.get(id).edges);
	}
	

	@Override
	public WeightedEdges getEdges(int id) {
		return new MapBasedWeighedEdes(id, nodes.get(id).edges);
	}

	@Override
	public boolean addUndirectedEdge(int id1, int id2, float weight) {
		boolean add1 = addDirectedEdge(id1, id2, weight);
		boolean add2 = addDirectedEdge(id2, id1, weight);
		return add1 && add2;
	}
	
	/**
	 * Add a directed edge between the two nodes.
	 * Return true if this edges did not exists before.
	 * 
	 * @param fromId
	 * @param toId
	 * @param weight
	 * @return
	 */
	private boolean addDirectedEdge(int fromId, int toId, float weight) {
		final NodeData nodeData = nodes.get(fromId);
		return (nodeData.edges.put(toId, weight) == null);
	}

	@Override
	public boolean removeNode(int id) {
		
		// remove all directed edges from this node to any other node
		final NodeData nodeData = nodes.remove(id);
		
		// if this nodes exists, remove all edges pointing to this node
		if(nodeData != null) {
			for(Integer otherId : nodeData.edges.keySet()) 
				nodes.get(otherId).edges.remove(id);
			return true;
		}
		return false;
	}


	@Override
	public boolean removeUndirectedEdge(int id1, int id2) {
		boolean remove1 = removeDirectedEdge(id1, id2);
		boolean remove2 = removeDirectedEdge(id2, id1);
		return remove1 && remove2;
	}
	
	/**
	 * Remove a undirected edge between the two nodes.
	 * Return true if any of the edges existed before.
	 * 
	 * @param fromId
	 * @param toId
	 * @return
	 */
	private boolean removeDirectedEdge(int fromId, int toId) {
		final NodeData nodeData = nodes.get(fromId);
		if(nodeData != null) 
			return (nodeData.edges.remove(toId) != null);
		return false;
	}
	

	@Override
	public float getEdgeWeight(int id1, int id2) {
		final NodeData nodeData = nodes.get(id1);
		if(nodeData == null)
			return 0;
		return nodeData.edges.get(id2);
	}

	@Override
	public void forEachEdge(WeightedEdgeConsumer consumer) {		
		for (Map.Entry<Integer, NodeData> nodeData : nodes.entrySet()) {
			final int nodeId = nodeData.getKey();			
			for (Map.Entry<Integer, Float> edge : nodeData.getValue().edges.entrySet()) {
				final int neighborId = edge.getKey();
				
				// implementation detail: all edges have a direction, ignore the edge in the other direction
				if(nodeId < neighborId) 
					consumer.accept(nodeId, neighborId, edge.getValue());
			}
		}
	}

	/**
	 * @see Serializable
	 * @param out
	 * @throws IOException
	 */
	public void writeObject(ObjectOutputStream out) throws IOException {

	}

	/**
	 * @see Serializable
	 * @param in
	 * @throws IOException
	 */
	public void readObject(ObjectInputStream in) throws IOException {

	}
	
	protected static class NodeData implements NodeView {
		
		protected final int label;
		protected final FeatureVector data;
		protected final Map<Integer,Float> edges;
		
		public NodeData(int label, FeatureVector data, int edgesPerNode) {
			this.label = label;
			this.data = data;
			this.edges = new HashMap<>(edgesPerNode);
		}

		@Override
		public int getLabel() {
			return label;
		}

		@Override
		public FeatureVector getFeature() {
			return data;
		}

		@Override
		public WeightedEdges getNeighbors() {
			return new MapBasedWeighedEdes(label, edges);
		}		
	}
	
	/**
	 * Contains immutable edge information for a single node.
	 *  
	 * @author Nico Hezel
	 */
	protected static class MapBasedWeighedEdes extends SetBasedIds implements WeightedEdges {
		
		protected final int refId;
		protected final Map<Integer, Float> idWeights;
		
		public MapBasedWeighedEdes(int refId, Map<Integer, Float> idWeights) {
			super(idWeights.keySet());
			this.refId = refId;
			this.idWeights = idWeights;
		}

		@Override
		public int getNodeId() {
			return refId;
		}
		
		@Override
		public void forEach(WeightedEdgeConsumer consumer) {
			final int id1 = this.refId;
			idWeights.forEach((id,weight) -> consumer.accept(id1, id, weight));
		}		
	}
	
	protected static class SetBasedIds implements NodeIds {
		
		protected final Set<Integer> ids;
		
		public SetBasedIds(Set<Integer> ids) {
			this.ids = ids;
		}

		@Override
		public int size() {
			return ids.size();
		}
		
		@Override
		public boolean contains(int id) {
			return ids.contains(id);
		}

		@Override
		public int[] toArray() {
			final int[] result = new int[size()];
			
			int counter = 0;
			final Iterator<Integer> it = this.ids.iterator();
			while(it.hasNext()) 
				result[counter++] = it.next();		
			
			return result;
		}	
		
		
		@Override
		public void forEach(IntConsumer consumer) {
			ids.forEach((id) -> consumer.accept(id));
		}
	}
}
