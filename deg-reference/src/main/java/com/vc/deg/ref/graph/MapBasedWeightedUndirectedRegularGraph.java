package com.vc.deg.ref.graph;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.IntConsumer;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.feature.FeatureFactory;
import com.vc.deg.graph.VertexIds;
import com.vc.deg.graph.VertexView;
import com.vc.deg.graph.WeightedEdgeConsumer;
import com.vc.deg.graph.WeightedEdges;
import com.vc.deg.io.LittleEndianDataInputStream;
import com.vc.deg.io.LittleEndianDataOutputStream;
import com.vc.deg.ref.feature.PrimitiveFeatureFactories;
import com.vc.deg.ref.search.ObjectDistance;





/**
 * This reference implementation uses the label of the data as its identifier for the nodes.
 * Every node contains of a label, data (stored in a MemoryView) and neighbors. The neighbors
 * are just the neighbor label and the edge weight.
 *  
 * @author Nico Hezel
 *
 */
public class MapBasedWeightedUndirectedRegularGraph {

	/**
	 * Label of the node is the identifier
	 */
	protected final Map<Integer, VertexData> nodes;
	
	/**
	 * The feature space knows who many bytes the node data (MemoryView) contains and how to compute the distance between two data objects.
	 */
	protected final FeatureSpace space;
	
	/**
	 * The number of edges per node is fixed and an even number
	 */
	protected final int edgesPerNode;

	public MapBasedWeightedUndirectedRegularGraph(int edgesPerNode, FeatureSpace space) {
		this.edgesPerNode = edgesPerNode;
		this.nodes = new HashMap<>();
		this.space = space;
	}

	public MapBasedWeightedUndirectedRegularGraph(int edgesPerNode, int expectedSize, FeatureSpace space) {
		this.edgesPerNode = edgesPerNode;
		this.nodes = new HashMap<>(expectedSize);	
		this.space = space;
	}
	
	public MapBasedWeightedUndirectedRegularGraph(int edgesPerNode, Map<Integer, VertexData> nodes, FeatureSpace space) {
		this.edgesPerNode = edgesPerNode;
		this.nodes = nodes;	
		this.space = space;
	}



	public FeatureSpace getFeatureSpace() {
		return space;
	}

	public VertexData getVertex(int id) {
		return nodes.getOrDefault(id, null);
	}

	public VertexData addVertex(int id, FeatureVector data) {
		if(hasNode(id) == false) {
			final VertexData newVertex = new VertexData(id, data, edgesPerNode+1);
			nodes.put(id, newVertex);
			return newVertex;
		}
		return null;
	}

	public int getVertexCount() {
		return nodes.size();
	}
	
	public Collection<VertexData> getVertices() {
		return nodes.values();
	}

	public boolean hasNode(int id) {
		return nodes.containsKey(id);
	}

	public boolean hasEdge(int id1, int id2) {
		final VertexData nodeData = nodes.get(id1);
		if(nodeData == null)
			return false;
		return nodeData.edges.containsKey(id2);
	}
	
	public int getEdgesPerVertex() {
		return edgesPerNode;
	}

//	public NodeIds getConnectedNodeIds(int id) {
//		return new MapBasedWeighedEdes(id, nodes.get(id).edges);
//	}
	
//	public WeightedEdges getEdges(int id) {
//		return new MapBasedWeighedEdes(id, nodes.get(id).edges);
//	}

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
		final VertexData nodeData = nodes.get(fromId);
		return (nodeData.edges.put(toId, weight) == null);
	}

	public VertexData removeNode(int id) {
		
		// remove all directed edges from this node to any other node
		final VertexData nodeData = nodes.remove(id);
		
		// if this nodes exists, remove all edges pointing to this node
		if(nodeData != null) {
			for(Integer otherId : nodeData.edges.keySet()) 
				nodes.get(otherId).edges.remove(id);
			return nodeData;
		}
		return null;
	}

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
		final VertexData nodeData = nodes.get(fromId);
		if(nodeData != null) 
			return (nodeData.edges.remove(toId) != null);
		return false;
	}
	
	/**
	 * 
	 * @param id1
	 * @param id2
	 * @return -1 if the edge does not exists
	 */
	public float getEdgeWeight(int id1, int id2) {
		final VertexData nodeData = nodes.get(id1);
		if(nodeData == null)
			return -1;
		final Float value = nodeData.edges.get(id2);
		if(value == null)
			return -1;
		return value;
	}
	
	public void forEachEdge(WeightedEdgeConsumer consumer) {		
		for (Map.Entry<Integer, VertexData> nodeData : nodes.entrySet()) {
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
     * Perform a search but stops when the toVertex was found.
     * 
	 * @param fromVertices
	 * @param toVertex
	 * @param k
	 * @param eps
	 * @return
	 */
	public List<ObjectDistance> hasPath(int[] fromVertices, int toVertex, int k, float eps) {
		final Map<Integer, ObjectDistance> trackback = new HashMap<>();
		final FeatureVector toVertexFeature = getVertex(toVertex).getFeature();

		// list of checked ids
		final Set<Integer> checkedIds = new HashSet<>(fromVertices.length + k*4);

		// items to traverse, start with the initial node
		final PriorityQueue<ObjectDistance> nextNodes = new PriorityQueue<>(k * 10); 
		for (int id : fromVertices) {
			if(checkedIds.add(id)) {
				final VertexData obj = getVertex(id);
				final float distance = space.computeDistance(toVertexFeature, obj.getFeature());
				nextNodes.add(new ObjectDistance(id, obj, distance));
				trackback.put(id, new ObjectDistance(id, obj, distance));
			}
		}

		// result set
		final TreeSet<ObjectDistance> results = new TreeSet<>(nextNodes);

		// search radius
		float radius = Float.MAX_VALUE;
		
		// iterate as long as good elements are in S
		while(nextNodes.size() > 0) {
			final ObjectDistance nextVertex = nextNodes.poll();

			// max distance reached
			if(nextVertex.getDistance() > radius * (1 + eps))
				break;

			// traverse never seen nodes
			for(Map.Entry<Integer, Float> edge : getVertex(nextVertex.getVertexId()).getEdges().entrySet()) {
				int neighborId = edge.getKey();
			
				if(checkedIds.add(neighborId)) {
					
				    // found our target node, create a path back to the entry node
			        if(neighborId == toVertex) {
			          final List<ObjectDistance> path = new ArrayList<>();
			          path.add(nextVertex);

			          int trackbackId = nextVertex.getLabel();
			          for (ObjectDistance lastVertex = trackback.get(trackbackId); lastVertex != null && trackbackId != lastVertex.getLabel(); trackbackId = lastVertex.getLabel(), lastVertex = trackback.get(trackbackId)) 
			        	  path.add(lastVertex);

			          return path;
			        }

					// follow this node further
			        final VertexData neighborVertex = getVertex(neighborId);
			        final float neighborDist = space.computeDistance(toVertexFeature, neighborVertex.getFeature());
					if(neighborDist <= radius * (1 + eps)) {
						final ObjectDistance candidate = new ObjectDistance(neighborId, neighborVertex, neighborDist);
						nextNodes.add(candidate);
				        trackback.put(neighborId, nextVertex);

						// remember the node
						if(neighborDist < radius) {
							results.add(candidate);
							if(results.size() > k) {
								results.pollLast();
								radius = results.last().getDistance();
							}							
						}
					}					
				}
			}
		}
		
		return new ArrayList<>();
	}
	
	/**
	 * 
	 * @param query
	 * @param k
	 * @param eps
	 * @param forbiddenIds TODO replace with filter
	 * @param entryPoints
	 * @return
	 */
	public TreeSet<ObjectDistance> search(FeatureVector query, int k, float eps, int[] forbiddenIds, int[] entryPoints) {
		
		// list of checked ids
		final Set<Integer> checkedIds = new HashSet<>(forbiddenIds.length + entryPoints.length + k*4);
		for (int id : forbiddenIds)
			checkedIds.add(id);
		
		// items to traverse, start with the initial node
		final PriorityQueue<ObjectDistance> nextNodes = new PriorityQueue<>(k * 10); 
		for (int id : entryPoints) {
			if(checkedIds.contains(id) == false) {
				checkedIds.add(id);
				final VertexData obj = getVertex(id);
				nextNodes.add(new ObjectDistance(id, obj, space.computeDistance(query, obj.getFeature())));
			}
		}

		// result set
		final TreeSet<ObjectDistance> results = new TreeSet<>(nextNodes);

		// search radius
		float radius = Float.MAX_VALUE;
		
		// iterate as long as good elements are in S
		while(nextNodes.size() > 0) {
			final ObjectDistance s = nextNodes.poll();

			// max distance reached
			if(s.getDistance() > radius * (1 + eps))
				break;

			// traverse never seen nodes
			for(Map.Entry<Integer, Float> edge : getVertex(s.getVertexId()).getEdges().entrySet()) {
				int neighborId = edge.getKey();
			
				if(checkedIds.add(neighborId)) {
					final VertexData n = getVertex(neighborId);
					final float nDist = space.computeDistance(query, n.getFeature());

					// follow this node further
					if(nDist <= radius * (1 + eps)) {
						final ObjectDistance candidate = new ObjectDistance(neighborId, n, nDist);
						nextNodes.add(candidate);

						// remember the node
						if(nDist < radius) {
							results.add(candidate);
							if(results.size() > k) {
								results.pollLast();
								radius = results.last().getDistance();
							}							
						}
					}					
				}
			}
		}
		
		return results;
	}

	/**
	 * 
	 * @param file
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public void writeToFile(Path file) throws ClassNotFoundException, IOException {
		Path tempFile = Paths.get(file.getParent().toString(), "~$" + file.getFileName().toString());

		try(BufferedOutputStream bos = new BufferedOutputStream(Files.newOutputStream(tempFile, TRUNCATE_EXISTING, CREATE, WRITE))) {
			// TODO support littleEndian files to be compatible with deglib in c-lang
			// https://github.com/google/guava/blob/master/guava/src/com/google/common/io/LittleEndianDataOutputStream.java
			// https://stackoverflow.com/questions/7024039/in-java-when-writing-to-a-file-with-dataoutputstream-how-do-i-define-the-endia
			DataOutput out = new LittleEndianDataOutputStream(bos);
			
			out.writeByte(space.metric());
			out.writeShort(space.dims());
			out.writeInt(getVertexCount());
			out.writeByte(edgesPerNode);
			
			for (int i = 0; i < getVertexCount(); i++) {
				VertexData node = this.getVertex(i);
				node.getFeature().writeObject(out);
				
				// get all edges in the graph, fill the remaining spots with self-loops and sort the result by the indices in ascending order
				List<IntFloat> edges = new ArrayList<>();	
				node.getEdges().forEach((neighborIdx, weight) -> {
					edges.add(new IntFloat(neighborIdx, weight));
				});
				for (int r = 0; r < edgesPerNode - node.getEdges().size(); r++)
					edges.add(new IntFloat(i, 0));
				edges.sort(IntFloat.asc());
				
				// write the data to the drive
				for (IntFloat edge : edges) 
					out.writeInt(edge.getIndex());
				for (IntFloat edge : edges) 
					out.writeFloat(edge.getValue());
				out.writeInt(node.getId());
			}
		}
		Files.move(tempFile, file, REPLACE_EXISTING);
	}
	
	public static MapBasedWeightedUndirectedRegularGraph readFromFile(Path file) throws IOException {
		String filename = file.getFileName().toString();
		int extStart = filename.lastIndexOf('.');
		int typeStart = filename.lastIndexOf('.', extStart-1);
		String dType = filename.substring(typeStart+1, extStart);
		return readFromFile(file, dType);
	}
	
	public static MapBasedWeightedUndirectedRegularGraph readFromFile(Path file, String featureType) throws IOException {
		try(BufferedInputStream bis = new BufferedInputStream(Files.newInputStream(file))) {
			final DataInput input = new LittleEndianDataInputStream(bis);

			// read meta data
			int metric = Byte.toUnsignedInt(input.readByte());
			int dims = Short.toUnsignedInt(input.readShort());
			long nodeCount = Integer.toUnsignedLong(input.readInt());
			int edgesPerNode = Byte.toUnsignedInt(input.readByte());
			
			// 	featureSize	=		     filesize - meta data - (edge data + label) * nodeCount   / nodeCount
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
			System.out.println("Read graph from file "+file.toString());
			final Map<Integer, VertexData> nodes = new HashMap<>((int)nodeCount); 
			for (int i = 0; i < nodeCount; i++) {
				
				// read the feature vector
				final FeatureVector feature = featureFactory.read(input);
				
				// read the edge data
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
				nodes.put(label, new VertexData(label, feature, edges));
				
				if(i % 100_000 == 0)
					System.out.println("Loaded "+i+" vertices");
			}
			System.out.println("Loaded "+nodeCount+" vertices");
			
			return new MapBasedWeightedUndirectedRegularGraph(edgesPerNode, nodes, space);
		}
	}
	
	
	/**
	 * @author Nico Hezel
	 * 
	 * TODO remove VertexView
	 */
	public static class VertexData implements VertexView {
		
		protected final int id;
		protected final FeatureVector data;
		protected final Map<Integer,Float> edges;
		
		public VertexData(int id, FeatureVector data, int edgesPerNode) {
			this.id = id;
			this.data = data;
			this.edges = new HashMap<>(edgesPerNode);
		}
		
		public VertexData(int id, FeatureVector data, Map<Integer,Float> edges) {
			this.id = id;
			this.data = data;
			this.edges = edges;
		}

		@Override
		public int getId() {
			return id;
		}

		@Override
		public FeatureVector getFeature() {
			return data;
		}

		// TODO remove VertexView
		@Override
		public WeightedEdges getNeighbors() {
			return new MapBasedWeighedEdes(id, edges);
		}	
		
		public Map<Integer,Float> getEdges() {
			return edges;
		}	
	}
	
	/**
	 * Contains edge information for a single node.
	 *  
	 * TODO remove VertexView
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
	 * TODO remove VertexView
	 * 
	 * @author Nico Hezel
	 */
	protected static class SetBasedIds implements VertexIds {
		
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

		@Override
		public Iterator<Integer> iterator() {
			return ids.iterator();
		}
	}
	
	/**
	 * Struct of int-float 
	 *
     * TODO move to common
     * 
	 * @author Nico Hezel
	 */
	protected static class IntFloat {
		private int num;
		private float val;

		public IntFloat(int number, float value) {
			num = number;
			val = value;
		}
		
		public int getIndex() {
			return num;
		}

		public float getValue() {
			return val;
		}
		
		@Override
		public String toString() {
			return "int: "+num+", val: "+val;
		}

		@Override
		public boolean equals(Object obj) {
			if(obj instanceof IntFloat)
				return ((IntFloat) obj).getIndex() == val;
			return super.equals(obj);
		}
		
		/**
		 * Order in ascending order using the index
		 *
		 * @return
		 */
		public static Comparator<IntFloat> asc() {
			return Comparator.comparingInt(IntFloat::getIndex).thenComparingDouble(IntFloat::getValue);
		}

		/**
		 * Order in descending order using the index
		 * 
		 * @return
		 */
		public static Comparator<IntFloat> desc() {
			return asc().reversed();
		}
	}
}