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
import java.util.BitSet;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.koloboke.collect.map.IntFloatMap;
import com.koloboke.collect.map.IntIntMap;
import com.koloboke.collect.map.hash.HashIntFloatMapFactory;
import com.koloboke.collect.map.hash.HashIntFloatMaps;
import com.koloboke.collect.map.hash.HashIntIntMapFactory;
import com.koloboke.collect.map.hash.HashIntIntMaps;
import com.koloboke.collect.set.IntSet;
import com.koloboke.collect.set.hash.HashIntSets;
import com.vc.deg.FeatureFactory;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.graph.GraphFilter;
import com.vc.deg.io.LittleEndianDataInputStream;
import com.vc.deg.io.LittleEndianDataOutputStream;
import com.vc.deg.ref.feature.PrimitiveFeatureFactories;

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
public class ArrayBasedWeightedUndirectedRegularGraph {

	private static Logger log = LoggerFactory.getLogger(ArrayBasedWeightedUndirectedRegularGraph.class);

	private static final HashIntFloatMapFactory intFloatMapFactory = HashIntFloatMaps.getDefaultFactory().withDefaultValue(Integer.MIN_VALUE);
	private static final HashIntIntMapFactory intIntMapFactory = HashIntIntMaps.getDefaultFactory().withDefaultValue(Integer.MIN_VALUE);

	
	/**
	 * Id of the vertex is the index
	 */
	protected final List<VertexData> vertices;
	
	/**
	 * Label to vertex id map
	 */
	protected final IntIntMap labelToId;
	
	/**
	 * The feature space knows who many bytes the vertex data contains and how to compute the distance between two data objects.
	 */
	protected final FeatureSpace space;
	
	/**
	 * The number of edges per vertex is fixed and an even number
	 */
	protected final int edgesPerVertex;

	public ArrayBasedWeightedUndirectedRegularGraph(int edgesPerVertex, FeatureSpace space) {
		this.edgesPerVertex = edgesPerVertex;
		this.vertices = new ArrayList<>();	
		this.labelToId = intIntMapFactory.newMutableMap();
		this.space = space;
	}

	public ArrayBasedWeightedUndirectedRegularGraph(int edgesPerVertex, int expectedSize, FeatureSpace space) {
		this.edgesPerVertex = edgesPerVertex;
		this.vertices = new ArrayList<>(expectedSize);	
		this.labelToId = intIntMapFactory.newMutableMap();
		this.space = space;
	}
	
	public ArrayBasedWeightedUndirectedRegularGraph(int edgesPerVertex, List<VertexData> vertices, IntIntMap labelToId, FeatureSpace space) {
		this.edgesPerVertex = edgesPerVertex;
		this.vertices = vertices;
		this.labelToId = labelToId;		
		this.space = space;
	}

	public FeatureSpace getFeatureSpace() {
		return space;
	}
	
	public int getVertexCount() {
		return vertices.size();
	}
	
	public Collection<VertexData> getVertices() {
		return Collections.unmodifiableCollection(vertices);
	}
	
	// ------------------------------------------------------------------------
	// -------------------------- label based methods -------------------------
	// ------------------------------------------------------------------------
		
	public VertexData getVertexByLabel(int label) {
		final int id = labelToId.getOrDefault(label, -1);
		return (id == -1) ? null : vertices.get(id);
	}

	public VertexData addVertex(int label, FeatureVector data) {
		if(hasVertex(label) == false) {
			final int vertexId = vertices.size();
			final VertexData newVertex = new VertexData(label, vertexId, data, edgesPerVertex+1);
			vertices.add(newVertex);
			labelToId.put(label, vertexId);
			return newVertex;
		}
		return null;
	}

	public boolean hasVertex(int label) {
		return labelToId.containsKey(label);
	}

	/**
	 * Deleted a vertex from the graph and update the edges list of all vertices pointing to the deleted vertex. 
	 * 
	 * @param label of the vertex to delete
	 * @return the edges of the deleted vertex
	 */
	public IntFloatMap removeVertexByLabel(int label) {
		
		// remove all directed edges from this vertex to any other vertex
		final int id = labelToId.remove(label);
		if(id != labelToId.defaultValue()) {
			final IntFloatMap removedVertexEdges = vertices.get(id).getEdges();
			
			// remove all edges pointing to this vertex
			for(int otherId : removedVertexEdges.keySet()) 
				vertices.get(otherId).getEdges().remove(id);
			
			// if the deleted label of the last vertex in the list, just remove it ...
			if(id == vertices.size() - 1) {
				vertices.remove(vertices.size() - 1);
			} else {
				
				// ... otherwise, move the last vertex in the list of vertices to the position of the deleted vertex
				final VertexData lastVertex = vertices.remove(vertices.size() - 1);
				vertices.set(id, new VertexData(lastVertex.getLabel(), id, lastVertex.getFeature(), lastVertex.getEdges()));
				labelToId.replace(lastVertex.getLabel(), id);
				
				// update the edge list of all the vertices pointing to the old position of last vertex
				for(int otherId : lastVertex.getEdges().keySet()) {
					final Map<Integer, Float> otherEdges = vertices.get(otherId).getEdges();
					otherEdges.put(id, otherEdges.remove(lastVertex.getId()));
				}
				
				// if last vertex is part of the edge list of the deleted vertex, update its id too
				if(removedVertexEdges.containsKey(lastVertex.getId())) 
					removedVertexEdges.put(id, removedVertexEdges.remove(lastVertex.getId()));
			} 
				
			return removedVertexEdges;
		}
		return null;
	}
	
	public Set<Integer> getVertexLabels() {
		return Collections.unmodifiableSet(labelToId.keySet());
	}
	
	// ------------------------------------------------------------------------
	// -------------------------- internal id based methods -------------------
	// ------------------------------------------------------------------------
	
	public VertexData getVertexById(int id) {
		return vertices.get(id);
	}

	public boolean hasEdge(int id1, int id2) {
		final VertexData vertexData = vertices.get(id1);
		if(vertexData == null)
			return false;
		return vertexData.getEdges().containsKey(id2);
	}
	
	public int getEdgesPerVertex() {
		return edgesPerVertex;
	}

	public boolean addUndirectedEdge(int id1, int id2, float weight) {		
		final boolean add1 = addDirectedEdge(id1, id2, weight);
		final boolean add2 = addDirectedEdge(id2, id1, weight);
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
		return (vertexData.getEdges().put(toId, weight) == vertexData.getEdges().defaultValue());
	}

	public boolean removeUndirectedEdge(int id1, int id2) {
		final boolean remove1 = removeDirectedEdge(id1, id2);
		final boolean remove2 = removeDirectedEdge(id2, id1);
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
			return (vertexData.getEdges().remove(toId) != vertexData.getEdges().defaultValue());
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
		return vertexData.getEdges().getOrDefault(id2, -1);
	}

	

	/**
     * Perform a search but stops when the toVertex was found.
     * 
	 * @param fromVertexIds
	 * @param toVertexId
	 * @param k
	 * @param eps
	 * @return
	 */
	public List<QueryDistance> hasPath(int[] fromVertexIds, int toVertexId, int k, float eps) {
		final Map<Integer, QueryDistance> trackback = new HashMap<>();
		final FeatureVector toVertexFeature = getVertexById(toVertexId).getFeature();

		// list of checked ids
		final BitSet checkedIds = new BitSet(getVertexCount());

		// items to traverse, start with the initial vertex
		final PriorityQueue<QueryDistance> nextVertexs = new PriorityQueue<>(k * 10); 
		for (int id : fromVertexIds) {
			if(checkedIds.get(id) == false) {
				checkedIds.set(id);
				
				final VertexData obj = getVertexById(id);
				final float distance = space.computeDistance(toVertexFeature, obj.getFeature());
				nextVertexs.add(new QueryDistance(toVertexId, toVertexFeature, obj, distance));
				trackback.put(id, new QueryDistance(toVertexId, toVertexFeature, obj, distance));
			}
		}

		// result set
		final TreeSet<QueryDistance> results = new TreeSet<>(nextVertexs);

		// search radius
		float radius = Float.MAX_VALUE;
		
		// iterate as long as good elements are in S
		while(nextVertexs.size() > 0) {
			final QueryDistance nextVertex = nextVertexs.poll();

			// max distance reached
			if(nextVertex.getDistance() > radius * (1 + eps))
				break;

			// traverse never seen vertices
			for(final int neighborId : nextVertex.getVertex().getEdges().keySet()) {
				if(checkedIds.get(neighborId) == false) {
					checkedIds.set(neighborId);
					
				    // found our target vertex, create a path back to the entry vertex
			        if(neighborId == toVertexId) {
			          final List<QueryDistance> path = new ArrayList<>();
			          path.add(nextVertex);

			          int trackbackId = nextVertex.getVertexId();
			          for (QueryDistance lastVertex = trackback.get(trackbackId); lastVertex != null && trackbackId != lastVertex.getVertexId(); trackbackId = lastVertex.getVertexId(), lastVertex = trackback.get(trackbackId)) 
			        	  path.add(lastVertex);

			          return path;
			        }

					// follow this vertex further
			        final VertexData neighborVertex = getVertexById(neighborId);
			        final float neighborDist = space.computeDistance(toVertexFeature, neighborVertex.getFeature());
					if(neighborDist <= radius * (1 + eps)) {
						final QueryDistance candidate = new QueryDistance(toVertexId, toVertexFeature, neighborVertex, neighborDist);
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
	 * @param queries
	 * @param k
	 * @param eps
	 * @param labelFilter of all valid ids. null disables the filter
	 * @param seedVertexIds
	 * @param allowSeedInResult
	 * @return
	 */
	public TreeSet<QueryDistance> search(Collection<FeatureVector> queries, int k, float eps, GraphFilter labelFilter, int[] seedVertexIds, boolean allowSeedInResult) {
		
		// allow all elements in the graph if the initial filter was null
		if(labelFilter == null)
			labelFilter = new GraphFilter.AllValidFilter(getVertexCount()); 
		
		// check the feature vector compatibility
		for (FeatureVector query : queries) {
			if(query.dims() != getFeatureSpace().dims() || query.getComponentType() != getFeatureSpace().getComponentType())
				throw new RuntimeException("Invalid query component type "+query.getComponentType().getSimpleName()+" or dimension "+query.dims()+
										   ", expected "+getFeatureSpace().getComponentType().getSimpleName()+" and "+getFeatureSpace().dims());
		}
		
		// list of checked ids
		final int vertexCount = getVertexCount();
		final BitSet checkedIds = new BitSet(vertexCount);
		
		// compute the min distance between the given fv and all the queries
		int distanceComputationCount = 0;
		final Function<VertexData, QueryDistance> calcMinDistance = (VertexData vertex) -> {			
			final FeatureVector fv = vertex.getFeature();
			
			int minDistIndex = -1;
			float minDistance = Float.MAX_VALUE;
			FeatureVector minDistQuery = null;
			
			int index = 0;
			for (FeatureVector query : queries) {				
				final float dist = space.computeDistance(query, fv);
				if(dist < minDistance) {
					minDistance = dist;
					minDistIndex = index;
					minDistQuery = query;
				}
				index++;
			}
			return new QueryDistance(minDistIndex, minDistQuery, vertex, minDistance);			
		};
	

		// result set
		final TreeSet<QueryDistance> results = new TreeSet<>();
		
		// search radius
		float radius = Float.MAX_VALUE;

		// if the filter only contains few valid ids brute force them all
		if(labelFilter.size() / vertexCount < 0.05) {
			
			// check all vertices of the graph if they pass the filter and are not a seed id if required 
			final IntSet seedIds = HashIntSets.newImmutableSet(seedVertexIds);			
			for (VertexData vertex : vertices) {				
				if(labelFilter.isValid(vertex.getLabel()) && (allowSeedInResult || seedIds.contains(vertex.getId()) == false)) {

					// keep all distances better than the worst in the result list
					distanceComputationCount++;
					final QueryDistance queryDistance = calcMinDistance.apply(vertex);
					if(queryDistance.getDistance() < radius) {
						results.add(queryDistance);

						// remove the last from the result list if the list gets to long
						if(results.size() > k) {
							results.pollLast();
							radius = results.last().getDistance();
						}
					}
				}
			}
			return results;
		}		
		
		
		// items to traverse, start with the initial vertex
		final PriorityQueue<QueryDistance> nextVertices = new PriorityQueue<>(k * 10); 
		for (int id : seedVertexIds) {
			if(checkedIds.get(id) == false) {
				checkedIds.set(id);
				nextVertices.add(calcMinDistance.apply(getVertexById(id)));
				distanceComputationCount++;
			}
		}

		// add seed vertices if they pass the filter
		if(allowSeedInResult) {
			for (QueryDistance queryDistance : nextVertices) {
				if(queryDistance.getDistance() < radius && labelFilter.isValid(queryDistance.getVertexLabel())) {
					results.add(queryDistance);
	
					if(results.size() > k) {
						results.pollLast();
						radius = results.last().getDistance();
					}
				}
			}
		}
		
		// iterate as long as good elements are in S
		while(nextVertices.size() > 0) {
			final QueryDistance nextVertex = nextVertices.poll();
			
			// eps goes to zero if distanceComputationCount gets close to 10% of the number of all vertices
			final float epsMod = eps * Math.max(1.f - (float)distanceComputationCount/(0.1f * vertexCount), 0.f); 

			// max distance reached
			if(nextVertex.getDistance() > radius * (1 + epsMod))
				break;

			// traverse never seen vertices
			for(final int neighborId : nextVertex.getVertex().getEdges().keySet()) {
				if(checkedIds.get(neighborId) == false) {
					checkedIds.set(neighborId);

					final VertexData neighbor = getVertexById(neighborId);
					final QueryDistance candidate = calcMinDistance.apply(neighbor);
					distanceComputationCount++;

					// follow this vertex further if its distance is better than the current radius
					final float nDist = candidate.getDistance();
					if(nDist <= radius * (1 + epsMod)) {
						
						// check the neighborhood of this node later
						nextVertices.add(candidate);

						 // remember the node, if its better than the worst in the result list and not forbidden
						if(nDist < radius && labelFilter.isValid(neighbor.getLabel())) {
							results.add(candidate);
							
							// update the search radius
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
	 * Create a deep copy of the graph. Only the Feature Vector will not be copied and their old reference will be used instead.
	 * 
	 * @return
	 */
	public ArrayBasedWeightedUndirectedRegularGraph copy() {
		final IntIntMap copyLabelMap = intIntMapFactory.newMutableMap(labelToId);
		final List<VertexData> copyVertices = new ArrayList<>(vertices.size());
		for (VertexData vertex : vertices) {
			final IntFloatMap copyEdges = intFloatMapFactory.newMutableMap(vertex.getEdges());
			copyVertices.add(new VertexData(vertex.getLabel(), vertex.getId(), vertex.getFeature(), copyEdges));
		}
		return new ArrayBasedWeightedUndirectedRegularGraph(edgesPerVertex, copyVertices, copyLabelMap, space);
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
			
			class IntFloat {
				public final int id;
				public final float distance;
				
				public IntFloat(int id, float distance) {
					this.id = id;
					this.distance = distance;
				}
				
				public int getId() {
					return id;
				}
				
				public float getDistance() {
					return distance;
				}
			}
			
			for (VertexData vertex : vertices) {
				vertex.getFeature().writeObject(out);
				
				// get the edges of the vertex, fill the remaining spots with self-loops and sort the result by the indices in ascending order
				final List<IntFloat> edges = new ArrayList<>();	
				vertex.getEdges().forEach((int neighborIdx, float weight) -> {
					edges.add(new IntFloat(neighborIdx, weight));
				});
				for (int r = 0; r < edgesPerVertex - vertex.getEdges().size(); r++)
					edges.add(new IntFloat( vertex.getId(), 0));
				edges.sort(Comparator.comparingInt(IntFloat::getId).thenComparingDouble(IntFloat::getDistance)); // sort ids low to high
				
				// write the data to the drive
				for (IntFloat edge : edges) 
					out.writeInt(edge.getId());
				for (IntFloat edge : edges) 
					out.writeFloat(edge.getDistance());
				out.writeInt(vertex.getLabel());
			}
		}
		Files.move(tempFile, file, REPLACE_EXISTING);
	}
	
	public static ArrayBasedWeightedUndirectedRegularGraph readFromFile(Path file) throws IOException {
		final String filename = file.getFileName().toString();
		final int extStart = filename.lastIndexOf('.');
		final int typeStart = filename.lastIndexOf('.', extStart-1);
		final String dType = filename.substring(typeStart+1, extStart);
		return readFromFile(file, dType);
	}
	
	public static ArrayBasedWeightedUndirectedRegularGraph readFromFile(Path file, String featureType) throws IOException {
		try(BufferedInputStream bis = new BufferedInputStream(Files.newInputStream(file))) {
			final DataInput input = new LittleEndianDataInputStream(bis);

			// read meta data
			final int metric = Byte.toUnsignedInt(input.readByte());
			final int dims = Short.toUnsignedInt(input.readShort());
			final long vertexCount = Integer.toUnsignedLong(input.readInt());
			final int edgesPerVertex = Byte.toUnsignedInt(input.readByte());
			
			// find the feature space specified in the file
			final FeatureSpace space = FeatureSpace.findFeatureSpace(featureType, metric, dims, false);
			if(space == null)
				throw new UnsupportedOperationException("No feature space found for featureType="+featureType+", metric="+metric+" and isNative=false");
			
			// empty graph
			if(vertexCount == 0) 
				return new ArrayBasedWeightedUndirectedRegularGraph(edgesPerVertex, space);
			
			// factory to create FeatureVectors based on the featureType
			FeatureFactory featureFactory = FeatureFactory.findFactory(featureType, dims);
			if(featureFactory == null)
				featureFactory = PrimitiveFeatureFactories.create(featureType, dims);
			if(featureFactory == null)
				throw new UnsupportedOperationException("No feature factory found for featureType="+featureType+" and dims="+dims);
			
			// 	featureSize	=		     filesize - meta data - (edge data   + label) * vertexCount   / vertexCount
			final int featureSize = (int)((Files.size(file) - 8 - ((edgesPerVertex * 8 + 4) * vertexCount)) / vertexCount);
			if(featureSize != featureFactory.featureSize())
				throw new UnsupportedOperationException("The feature factory for featureType="+featureType+" and dims="+dims+" produces features with "+featureFactory.featureSize()+" bytes but the graph contains features with "+featureSize+" bytes.");
			if(featureSize != space.featureSize())
				throw new UnsupportedOperationException("The feature space for featureType="+featureType+", metric="+metric+" and isNative=false expects features with "+space.featureSize()+" bytes but the graph contains features with "+featureSize+" bytes.");
			
			// the references implementation uses 
			if(vertexCount > Integer.MAX_VALUE)
				throw new UnsupportedOperationException("The reference implementation does not allow graphs with more than "+Integer.MAX_VALUE+" vertices");
			
			// read the vertex data
			log.debug("Read graph from file "+file.toString());
			final IntIntMap labelMap = intIntMapFactory.newMutableMap((int)vertexCount); 
			final List<VertexData> vertices = new ArrayList<>((int)vertexCount); 
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
				final IntFloatMap edges = intFloatMapFactory.newMutableMap(edgesPerVertex);
				for (int e = 0; e < edgesPerVertex; e++)
					edges.put(neighborIds[e], weights[e]);
				
				// read the label
				final int label = input.readInt();
				
				// create the vertex data
				labelMap.put(label, i);
				vertices.add(new VertexData(label, i, feature, edges));
				
				if(i % 100_000 == 0)
					log.debug("Loaded "+i+" vertices");
			}
			log.debug("Loaded "+vertexCount+" vertices");
			
			return new ArrayBasedWeightedUndirectedRegularGraph(edgesPerVertex, vertices, labelMap, space);
		}
	}
}