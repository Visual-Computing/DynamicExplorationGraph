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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeSet;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.feature.FeatureFactory;
import com.vc.deg.io.LittleEndianDataInputStream;
import com.vc.deg.io.LittleEndianDataOutputStream;
import com.vc.deg.ref.feature.PrimitiveFeatureFactories;
import com.vc.deg.ref.search.ObjectDistance;

/**
 * A weighted undirected regular graph based on HashMaps to store the vertices and edges.
 * 
 * This reference implementation uses the label of the data as its identifier for the vertices.
 * Every vertex contains of a label, a feature vector and a set of edges. 
 * Each edge know the neighbor label and the edge weight.
 *  
 * @author Nico Hezel
 *
 */
public class MapBasedWeightedUndirectedRegularGraph {

	/**
	 * Label of the vertex is the identifier
	 */
	protected final Map<Integer, VertexData> vertices;
	
	/**
	 * The feature space knows who many bytes the vertex data contains and how to compute the distance between two data objects.
	 */
	protected final FeatureSpace space;
	
	/**
	 * The number of edges per vertex is fixed and an even number
	 */
	protected final int edgesPerVertex;

	public MapBasedWeightedUndirectedRegularGraph(int edgesPerVertex, FeatureSpace space) {
		this.edgesPerVertex = edgesPerVertex;
		this.vertices = new HashMap<>();
		this.space = space;
	}

	public MapBasedWeightedUndirectedRegularGraph(int edgesPerVertex, int expectedSize, FeatureSpace space) {
		this.edgesPerVertex = edgesPerVertex;
		this.vertices = new HashMap<>(expectedSize);	
		this.space = space;
	}
	
	public MapBasedWeightedUndirectedRegularGraph(int edgesPerVertex, Map<Integer, VertexData> vertices, FeatureSpace space) {
		this.edgesPerVertex = edgesPerVertex;
		this.vertices = vertices;	
		this.space = space;
	}

	public FeatureSpace getFeatureSpace() {
		return space;
	}

	public VertexData getVertex(int id) {
		return vertices.getOrDefault(id, null);
	}

	public VertexData addVertex(int id, FeatureVector data) {
		if(hasVertex(id) == false) {
			final VertexData newVertex = new VertexData(id, data, edgesPerVertex+1);
			vertices.put(id, newVertex);
			return newVertex;
		}
		return null;
	}

	public int getVertexCount() {
		return vertices.size();
	}
	
	public Collection<Integer> getVertexIds() {
		return Collections.unmodifiableCollection(vertices.keySet());
	}
	
	public Collection<VertexData> getVertices() {
		return Collections.unmodifiableCollection(vertices.values());
	}

	public boolean hasVertex(int id) {
		return vertices.containsKey(id);
	}

	public boolean hasEdge(int id1, int id2) {
		final VertexData vertexData = vertices.get(id1);
		if(vertexData == null)
			return false;
		return vertexData.edges.containsKey(id2);
	}
	
	public int getEdgesPerVertex() {
		return edgesPerVertex;
	}

	public boolean addUndirectedEdge(int id1, int id2, float weight) {		
		boolean add1 = addDirectedEdge(id1, id2, weight);
		boolean add2 = addDirectedEdge(id2, id1, weight);
		return add1 && add2;
	}
	
	/**
	 * Add a directed edge between the two vertices.
	 * Return true if this edges did not exists before.
	 * 
	 * @param fromId
	 * @param toId
	 * @param weight
	 * @return
	 */
	private boolean addDirectedEdge(int fromId, int toId, float weight) {
		final VertexData vertexData = vertices.get(fromId);
		return (vertexData.edges.put(toId, weight) == null);
	}

	public VertexData removeVertex(int id) {
		
		// remove all directed edges from this vertex to any other vertex
		final VertexData vertexData = vertices.remove(id);
		
		// if this vertices exists, remove all edges pointing to this vertex
		if(vertexData != null) {
			for(Integer otherId : vertexData.edges.keySet()) 
				vertices.get(otherId).edges.remove(id);
			return vertexData;
		}
		return null;
	}

	public boolean removeUndirectedEdge(int id1, int id2) {
		boolean remove1 = removeDirectedEdge(id1, id2);
		boolean remove2 = removeDirectedEdge(id2, id1);
		return remove1 && remove2;
	}
	
	/**
	 * Remove a undirected edge between the two vertices.
	 * Return true if any of the edges existed before.
	 * 
	 * @param fromId
	 * @param toId
	 * @return
	 */
	private boolean removeDirectedEdge(int fromId, int toId) {
		final VertexData vertexData = vertices.get(fromId);
		if(vertexData != null) 
			return (vertexData.edges.remove(toId) != null);
		return false;
	}
	
	/**
	 * 
	 * @param id1
	 * @param id2
	 * @return -1 if the edge does not exists
	 */
	public float getEdgeWeight(int id1, int id2) {
		final VertexData vertexData = vertices.get(id1);
		if(vertexData == null)
			return -1;
		final Float value = vertexData.edges.get(id2);
		if(value == null)
			return -1;
		return value;
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

		// items to traverse, start with the initial vertex
		final PriorityQueue<ObjectDistance> nextVertexs = new PriorityQueue<>(k * 10); 
		for (int id : fromVertices) {
			if(checkedIds.add(id)) {
				final VertexData obj = getVertex(id);
				final float distance = space.computeDistance(toVertexFeature, obj.getFeature());
				nextVertexs.add(new ObjectDistance(id, distance));
				trackback.put(id, new ObjectDistance(id, distance));
			}
		}

		// result set
		final TreeSet<ObjectDistance> results = new TreeSet<>(nextVertexs);

		// search radius
		float radius = Float.MAX_VALUE;
		
		// iterate as long as good elements are in S
		while(nextVertexs.size() > 0) {
			final ObjectDistance nextVertex = nextVertexs.poll();

			// max distance reached
			if(nextVertex.getDistance() > radius * (1 + eps))
				break;

			// traverse never seen vertices
			for(Map.Entry<Integer, Float> edge : getVertex(nextVertex.getLabel()).getEdges().entrySet()) {
				int neighborId = edge.getKey();
			
				if(checkedIds.add(neighborId)) {
					
				    // found our target vertex, create a path back to the entry vertex
			        if(neighborId == toVertex) {
			          final List<ObjectDistance> path = new ArrayList<>();
			          path.add(nextVertex);

			          int trackbackId = nextVertex.getLabel();
			          for (ObjectDistance lastVertex = trackback.get(trackbackId); lastVertex != null && trackbackId != lastVertex.getLabel(); trackbackId = lastVertex.getLabel(), lastVertex = trackback.get(trackbackId)) 
			        	  path.add(lastVertex);

			          return path;
			        }

					// follow this vertex further
			        final VertexData neighborVertex = getVertex(neighborId);
			        final float neighborDist = space.computeDistance(toVertexFeature, neighborVertex.getFeature());
					if(neighborDist <= radius * (1 + eps)) {
						final ObjectDistance candidate = new ObjectDistance(neighborId, neighborDist);
						nextVertexs.add(candidate);
				        trackback.put(neighborId, nextVertex);

						// remember the vertex
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
		
		// items to traverse, start with the initial vertex
		final PriorityQueue<ObjectDistance> nextVertices = new PriorityQueue<>(k * 10); 
		for (int id : entryPoints) {
			if(checkedIds.contains(id) == false) {
				checkedIds.add(id);
				final VertexData obj = getVertex(id);
				nextVertices.add(new ObjectDistance(id, space.computeDistance(query, obj.getFeature())));
			}
		}

		// result set
		final TreeSet<ObjectDistance> results = new TreeSet<>(nextVertices);

		// search radius
		float radius = Float.MAX_VALUE;
		
		// iterate as long as good elements are in S
		while(nextVertices.size() > 0) {
			final ObjectDistance nextVertex = nextVertices.poll();

			// max distance reached
			if(nextVertex.getDistance() > radius * (1 + eps))
				break;

			// traverse never seen vertices
			for(Map.Entry<Integer, Float> edge : getVertex(nextVertex.getLabel()).getEdges().entrySet()) {
				int neighborId = edge.getKey();
			
				if(checkedIds.add(neighborId)) {
					final VertexData n = getVertex(neighborId);
					final float nDist = space.computeDistance(query, n.getFeature());

					// follow this vertex further
					if(nDist <= radius * (1 + eps)) {
						final ObjectDistance candidate = new ObjectDistance(neighborId, nDist);
						nextVertices.add(candidate);

						// remember the vertex
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
	 * @param query
	 * @param k
	 * @param eps
	 * @param forbiddenIds TODO replace with filter
	 * @param entryPoints
	 * @return
	 */
	public TreeSet<ObjectDistance> explore(int entryVertex, int k, int maxDistanceComputationCount, int[] forbiddenIds) {
	    int distanceComputationCount = 0;

		// list of checked ids
		final Set<Integer> checkedIds = new HashSet<>(forbiddenIds.length + k*2);
		for (int id : forbiddenIds)
			checkedIds.add(id);
		
		// items to traverse, start with the initial vertex
		final PriorityQueue<ObjectDistance> nextVertices = new PriorityQueue<>(k * 2); 
		checkedIds.add(entryVertex);
		nextVertices.add(new ObjectDistance(entryVertex, 0));
		final FeatureVector query = getVertex(entryVertex).getFeature();

		// result set
		final TreeSet<ObjectDistance> results = new TreeSet<>(nextVertices);

		// search radius
		float radius = Float.MAX_VALUE;
		
		// iterate as long as good elements are in S
		while(nextVertices.size() > 0) {
			final ObjectDistance nextVertex = nextVertices.poll();

			// max distance reached
			if(nextVertex.getDistance() > radius)
				break;

			// traverse never seen vertices
			for(Map.Entry<Integer, Float> edge : getVertex(nextVertex.getLabel()).getEdges().entrySet()) {
				final int neighborId = edge.getKey();
			
				if(checkedIds.add(neighborId)) {
					final VertexData n = getVertex(neighborId);
					final float nDist = space.computeDistance(query, n.getFeature());

					// follow this vertex further if its distance is better than the current radius
					if(nDist < radius) {
						
						// check the neighborhood of this node later
						final ObjectDistance candidate = new ObjectDistance(neighborId, nDist);
						nextVertices.add(candidate);

						 // remember the node, if its better than the worst in the result list
						results.add(candidate);
						
						// update the search radius
						if(results.size() > k) {
							results.pollLast();
							radius = results.last().getDistance();
						}	
					}
					
				      // early stop after to many computations
			        if(++distanceComputationCount >= maxDistanceComputationCount)
			          return results;
				}
			}
		}
		
		return results;
	}
	
	/**
	 * Create a deep copy of the graph. Only the Feature Vector will not be copied and their old reference will be used instead.
	 * 
	 * @return
	 */
	public MapBasedWeightedUndirectedRegularGraph copy() {
		final Map<Integer, VertexData> copyVertices = new HashMap<>(vertices.size());
		for (Map.Entry<Integer, VertexData> entry : vertices.entrySet()) {
			final VertexData vertex = entry.getValue();
			final Map<Integer, Float> copyEdges = new HashMap<>(vertex.getEdges());
			copyVertices.put(entry.getKey(), new VertexData(entry.getKey(), vertex.getFeature(), copyEdges));
		}
		return new MapBasedWeightedUndirectedRegularGraph(edgesPerVertex, copyVertices, space);
	}

	/**
	 * 
	 * @param file
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public void writeToFile(Path file) throws ClassNotFoundException, IOException {
		final Path tempFile = Paths.get(file.getParent().toString(), "~$" + file.getFileName().toString());

		try(BufferedOutputStream bos = new BufferedOutputStream(Files.newOutputStream(tempFile, TRUNCATE_EXISTING, CREATE, WRITE))) {

			// https://github.com/google/guava/blob/master/guava/src/com/google/common/io/LittleEndianDataOutputStream.java
			// https://stackoverflow.com/questions/7024039/in-java-when-writing-to-a-file-with-dataoutputstream-how-do-i-define-the-endia
			final DataOutput out = new LittleEndianDataOutputStream(bos);
			
			out.writeByte(space.metric());
			out.writeShort(space.dims());
			out.writeInt(getVertexCount());
			out.writeByte(edgesPerVertex);
			
			for (VertexData vertex : vertices.values()) {
				vertex.getFeature().writeObject(out);
				
				// get all edges in the graph, fill the remaining spots with self-loops and sort the result by the indices in ascending order
				final List<ObjectDistance> edges = new ArrayList<>();	
				vertex.getEdges().forEach((neighborIdx, weight) -> {
					edges.add(new ObjectDistance(neighborIdx, weight));
				});
				for (int r = 0; r < edgesPerVertex - vertex.getEdges().size(); r++)
					edges.add(new ObjectDistance(vertex.getId(), 0));
				edges.sort(ObjectDistance.ascByIndex());
				
				// write the data to the drive
				for (ObjectDistance edge : edges) 
					out.writeInt(edge.getLabel());
				for (ObjectDistance edge : edges) 
					out.writeFloat(edge.getDistance());
				out.writeInt(vertex.getId());
			}
		}
		Files.move(tempFile, file, REPLACE_EXISTING);
	}
	
	public static MapBasedWeightedUndirectedRegularGraph readFromFile(Path file) throws IOException {
		final String filename = file.getFileName().toString();
		final int extStart = filename.lastIndexOf('.');
		final int typeStart = filename.lastIndexOf('.', extStart-1);
		final String dType = filename.substring(typeStart+1, extStart);
		return readFromFile(file, dType);
	}
	
	public static MapBasedWeightedUndirectedRegularGraph readFromFile(Path file, String featureType) throws IOException {
		try(BufferedInputStream bis = new BufferedInputStream(Files.newInputStream(file))) {
			final DataInput input = new LittleEndianDataInputStream(bis);

			// read meta data
			int metric = Byte.toUnsignedInt(input.readByte());
			int dims = Short.toUnsignedInt(input.readShort());
			long vertexCount = Integer.toUnsignedLong(input.readInt());
			int edgesPerVertex = Byte.toUnsignedInt(input.readByte());
			
			// 	featureSize	=		     filesize - meta data - (edge data + label) * vertexCount   / vertexCount
			int featureSize = (int)((Files.size(file) - 8 - ((edgesPerVertex * 8 + 4) * vertexCount)) / vertexCount);

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
			if(vertexCount > Integer.MAX_VALUE)
				throw new UnsupportedOperationException("The reference implementation does not allow graphs with more than "+Integer.MAX_VALUE+" vertices");
			
			// read the vertex data
			System.out.println("Read graph from file "+file.toString());
			final Map<Integer, VertexData> vertices = new HashMap<>((int)vertexCount); 
			for (int i = 0; i < vertexCount; i++) {
				
				// read the feature vector
				final FeatureVector feature = featureFactory.read(input);
				
				// read the edge data
				final int[] neighborIds = new int[edgesPerVertex];
				for (int e = 0; e < edgesPerVertex; e++) 
					neighborIds[e] = input.readInt();
				final float[] weights = new float[edgesPerVertex];
				for (int e = 0; e < edgesPerVertex; e++) 
					weights[e] = input.readFloat();	
				Map<Integer,Float> edges = new HashMap<>(edgesPerVertex);
				for (int e = 0; e < edgesPerVertex; e++)
					edges.put(neighborIds[e], weights[e]);
				
				// read the label
				int label = input.readInt();
				
				// create the vertex data
				vertices.put(label, new VertexData(label, feature, edges));
				
				if(i % 100_000 == 0)
					System.out.println("Loaded "+i+" vertices");
			}
			System.out.println("Loaded "+vertexCount+" vertices");
			
			return new MapBasedWeightedUndirectedRegularGraph(edgesPerVertex, vertices, space);
		}
	}
	
	
	/**
	 * @author Nico Hezel
	 */
	public static class VertexData  {
		
		protected final int id;
		protected final FeatureVector data;
		protected final Map<Integer,Float> edges;
		
		public VertexData(int id, FeatureVector data, int edgesPerVertex) {
			this.id = id;
			this.data = data;
			this.edges = new HashMap<>(edgesPerVertex);
		}
		
		public VertexData(int id, FeatureVector data, Map<Integer,Float> edges) {
			this.id = id;
			this.data = data;
			this.edges = edges;
		}

		public int getId() {
			return id;
		}

		public FeatureVector getFeature() {
			return data;
		}
		
		public Map<Integer,Float> getEdges() {
			return edges;
		}	
	}
}