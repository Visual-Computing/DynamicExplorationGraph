package com.vc.deg.graph;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.function.IntConsumer;

import com.vc.deg.graph.NodeIds;
import com.vc.deg.graph.WeightedEdgeConsumer;
import com.vc.deg.graph.WeightedEdges;
import com.vc.deg.graph.WeightedUndirectedGraph;




/**
 * Der Interne Graph mit seinen addNode,deleteNode,addEdge,deleteEdge,serialize Funktionen.
 * Stelle keine Such oder Navigationsmöglichen.
 * 
 * @author Neiko
 *
 */
public class EvenRegularWeightedUndirectedGraph implements Serializable, WeightedUndirectedGraph {

	private static final long serialVersionUID = -9100450642805380353L;

	protected final Map<Integer, Map<Integer,Float>> nodes;
	protected int edgesPerNode;

	public EvenRegularWeightedUndirectedGraph(int edgesPerNode) {
		this.edgesPerNode = edgesPerNode;
		this.nodes = new HashMap<>();
	}

	public EvenRegularWeightedUndirectedGraph(int edgesPerNode, int expectedSize) {
		this.edgesPerNode = edgesPerNode;
		this.nodes = new HashMap<>(expectedSize);		
	}

	/**
	 * Add and return a new node or return an existing one.
	 * 
	 * @param id
	 * @return
	 */
	protected Map<Integer,Float> getOrAddNode(int id) {
		Map<Integer,Float> edges = nodes.getOrDefault(id, null);
		if(edges == null) {
			edges = new HashMap<>();
			nodes.put(id, edges);
		}				  
		return edges;
	}
	
	@Override
	public boolean addNode(int id) {
		if(hasNode(id) == false) {
			nodes.put(id, new HashMap<>(edgesPerNode+1));
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
		final Map<Integer,Float> edges = nodes.get(id1);
		if(edges == null)
			return false;
		return edges.containsKey(id2);
	}

	@Override
	public NodeIds getConnectedNodeIds(int id) {
		return new MapBasedWeighedEdes(id, nodes.get(id));
	}
	

	@Override
	public WeightedEdges getEdges(int id) {
		return new MapBasedWeighedEdes(id, nodes.get(id));
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
		final Map<Integer,Float> edges = getOrAddNode(fromId);
		return (edges.put(toId, weight) == null);
	}

	@Override
	public boolean removeNode(int id) {
		
		// remove all directed edges from this node to any other node
		final Map<Integer,Float> edges = nodes.remove(id);
		
		// if this nodes exists, remove all edges pointing to this node
		if(edges != null) {
			for(Integer otherId : edges.keySet()) 
				nodes.get(otherId).remove(id);
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
		final Map<Integer,Float> edges = nodes.get(fromId);
		if(edges != null) 
			return (edges.remove(toId) != null);
		return false;
	}
	

	@Override
	public float getEdgeWeight(int id1, int id2) {
		final Map<Integer,Float> edges = nodes.get(id1);
		if(edges == null)
			return 0;
		return edges.get(id2);
	}

	@Override
	public void forEachEdge(WeightedEdgeConsumer consumer) {		
		for (Map.Entry<Integer, Map<Integer, Float>> edges : nodes.entrySet()) {
			final int nodeId = edges.getKey();			
			for (Map.Entry<Integer, Float> edge : edges.getValue().entrySet()) {
				final int neighborId = edge.getKey();
				
				// implementation detail: all edges have a direction, ignore the edge in the other direction
				if(nodeId < neighborId) 
					consumer.accept(nodeId, neighborId, edge.getValue());
			}
		}
	}

	public void writeObject(ObjectOutputStream out) throws IOException {

	}

	public void readObject(ObjectInputStream in) throws IOException {

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
