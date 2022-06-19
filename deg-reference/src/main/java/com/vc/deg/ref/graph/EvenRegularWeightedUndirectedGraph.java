package com.vc.deg.ref.graph;

import java.io.BufferedInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.function.IntConsumer;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.feature.FeatureFactory;
import com.vc.deg.graph.NodeIds;
import com.vc.deg.graph.NodeView;
import com.vc.deg.graph.WeightedEdgeConsumer;
import com.vc.deg.graph.WeightedEdges;
import com.vc.deg.graph.WeightedUndirectedGraph;
import com.vc.deg.io.LittleEndianDataInputStream;
import com.vc.deg.ref.feature.PrimitiveFeatureFactories;




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
public class EvenRegularWeightedUndirectedGraph implements WeightedUndirectedGraph {

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
	
	public EvenRegularWeightedUndirectedGraph(int edgesPerNode, Map<Integer, NodeData> nodes, FeatureSpace space) {
		this.edgesPerNode = edgesPerNode;
		this.nodes = nodes;	
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
	public void writeObject(DataOutputStream out) throws IOException {
		// TODO features need to be written with the help of the columnBasedFFeatureFactory in order to store only the data and no size
		
		// TODO the neighbor ids need to be stored in ascending order
	}
	
	public static EvenRegularWeightedUndirectedGraph readFromFile(Path file) throws IOException {
		String filename = file.getFileName().toString();
		int extStart = filename.lastIndexOf('.');
		int typeStart = filename.lastIndexOf('.', extStart-1);
		String dType = filename.substring(typeStart+1, extStart);
		return readFromFile(file, dType);
	}
	
	public static EvenRegularWeightedUndirectedGraph readFromFile(Path file, String featureType) throws IOException {
		try(InputStream is = Files.newInputStream(file)) {
			BufferedInputStream bis = new BufferedInputStream(is);
			LittleEndianDataInputStream input = new LittleEndianDataInputStream(bis);

			// read meta data
			int metric = Byte.toUnsignedInt(input.readByte());
			int dims = Short.toUnsignedInt(input.readShort());
			long nodeCount = Integer.toUnsignedLong(input.readInt());
			int edgesPerNode = Byte.toUnsignedInt(input.readByte());
			
			// 	featureSize	=		    filesize - meta data - (edge data + label) * nodeCount  / nodeCount
			int featureSize = (int)((Files.size(file) - 8 - ((edgesPerNode * 8 + 4) * nodeCount)) / nodeCount);

			// factory to create FeatureVectors based on the featureType
			FeatureFactory featureFactory = PrimitiveFeatureFactories.get(featureType, dims);
			if(featureFactory == null)
				featureFactory = FeatureFactory.findFactory(featureType, dims);
			if(featureFactory == null)
				throw new UnsupportedOperationException("No feature factory found for featureType="+featureType+" and dims="+dims);
			if(featureSize != featureFactory.featureSize())
				throw new UnsupportedOperationException("The feature factory for featureType="+featureType+" and dims="+dims+" produces features with "+featureFactory.featureSize()+" bytes but the graph contains features with "+featureSize+" bytes.");
		
			// find the feature space specified in the file
			FeatureSpace space = FeatureSpace.findFeatureSpace(featureType, metric, false);
			if(space == null)
				throw new UnsupportedOperationException("No feature space found for featureType="+featureType+", metric="+metric+" and isNative=false");
			if(featureSize != space.featureSize())
				throw new UnsupportedOperationException("The feature space for featureType="+featureType+", metric="+metric+" and isNative=false expects features with "+space.featureSize()+" bytes but the graph contains features with "+featureSize+" bytes.");
			if(dims != space.dims())
				throw new UnsupportedOperationException("The feature space for featureType="+featureType+", metric="+metric+" and isNative=false expects features with "+space.dims()+" dimensions but the graph contains features with "+dims+" dimensions.");
			
			// the references implementation uses 
			if(nodeCount > Integer.MAX_VALUE)
				throw new UnsupportedOperationException("The reference implementation does not allow graphs with more than "+Integer.MAX_VALUE+" nodes");
			
			// read the node data
			final Map<Integer, NodeData> nodes = new HashMap<>((int)nodeCount); 
			for (int i = 0; i < nodeCount; i++) {			
				
				// read the feature vector
				final FeatureVector feature = featureFactory.read(input);
				
				// read the edge dta
				final int[] neighborIds = new int[edgesPerNode];
				for (int e = 0; e < edgesPerNode; e++) 
					neighborIds[e] = input.readInt();
				final float[] weights = new float[edgesPerNode];
				for (int e = 0; e < edgesPerNode; e++) 
					weights[e] = input.readFloat();				
				Map<Integer,Float> edges = new HashMap<>(edgesPerNode);
				for (int e = 0; e < edgesPerNode; e++)
					edges.put(neighborIds[e], weights[e]);
				
				// read the label
				int label = input.readInt();
				
				// create the node data
				nodes.put(label, new NodeData(label, feature, edges));
				
				if(i % 100_000 == 0)
					System.out.println("loaded "+i+" nodes");
			}
			
			return new EvenRegularWeightedUndirectedGraph(edgesPerNode, nodes, space);
		}
	}
	
	
	/**
	 * @author Nico Hezel
	 */
	protected static class NodeData implements NodeView {
		
		protected final int label;
		protected final FeatureVector data;
		protected final Map<Integer,Float> edges;
		
		public NodeData(int label, FeatureVector data, int edgesPerNode) {
			this.label = label;
			this.data = data;
			this.edges = new HashMap<>(edgesPerNode);
		}
		
		public NodeData(int label, FeatureVector data, Map<Integer,Float> edges) {
			this.label = label;
			this.data = data;
			this.edges = edges;
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
	 * Contains edge information for a single node.
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
	
	/**
	 * Set of node ids
	 * 
	 * @author Nico Hezel
	 */
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
