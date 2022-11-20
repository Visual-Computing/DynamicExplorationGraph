package com.vc.deg.ref.designer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.stream.Stream;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.graph.GraphDesigner;
import com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph;
import com.vc.deg.ref.graph.VertexData;
import com.vc.deg.ref.graph.VertexDistance;


/**
 * Verbessert und erweitert den Graphen
 * 
 * @author Nico Hezel
 *
 */
public class EvenRegularGraphDesigner implements GraphDesigner {

	protected final ArrayBasedWeightedUndirectedRegularGraph graph;
	
	protected final AtomicLong manipulationCounter;
	protected final ConcurrentLinkedQueue<BuilderAddTask> newEntryQueue;
	protected final ConcurrentLinkedQueue<BuilderRemoveTask> removeEntryQueue;
	protected boolean stopBuilding;

	protected Random rnd;

	
	// hyper parameters
	protected int extendK;
	protected float extendEps;

	protected int improveK;
	protected float improveEps;
	
	protected int maxPathLength;

	public EvenRegularGraphDesigner(ArrayBasedWeightedUndirectedRegularGraph graph) {
		this.graph = graph;
		this.rnd = new Random(7);
		this.manipulationCounter = new AtomicLong(0);
		this.newEntryQueue = new ConcurrentLinkedQueue<>();
		this.removeEntryQueue = new ConcurrentLinkedQueue<>();

		// default hyper parameters
		setExtendK(graph.getEdgesPerVertex() * 2);
		setExtendEps(0.2f);
		setImproveK(graph.getEdgesPerVertex());
		setImproveEps(0.001f);
		setMaxPathLength(5);
	}



	// ----------------------------------------------------------------------------------------------
	// ----------------------------------- hyper parameters -----------------------------------------
	// ----------------------------------------------------------------------------------------------

	@Override
	public void setRandom(Random rnd) {
		this.rnd = rnd;
	}

	@Override
	public void setExtendK(int k) {
		this.extendK = k;
	}

	@Override
	public void setExtendEps(float eps) {
		this.extendEps = eps;
	}

	@Override
	public void setImproveK(int k) {
		this.improveK = k;
	}

	@Override
	public void setImproveEps(float eps) {
		this.improveEps = eps;
	}
	
	@Override
	public void setMaxPathLength(int maxPathLength) {
		this.maxPathLength = maxPathLength;
	}



	// ----------------------------------------------------------------------------------------------
	// ----------------------------------- evaluation methods ---------------------------------------
	// ----------------------------------------------------------------------------------------------

	@Override
	public float calcAvgEdgeWeight() {
		double sum = 0;
		int count = 0;
		for(VertexData data : graph.getVertices()) {
			final Collection<Float> weights = data.getEdges().values();
			for(float weight : weights) 
				sum += weight;
			count += weights.size();
		}
		return (float)(sum/count);
	}

	@Override
	public boolean checkGraphValidation(int expectedVertices, int expectedNeighbors) {

		// check vertex count
		if(graph.getVertexCount() != expectedVertices)				
			throw new RuntimeException("The graph has an unexpected number of vertices. expected "+expectedVertices+" got "+graph.getVertexCount());

		final int targetNeighborCount = Math.min(graph.getVertexCount() - 1, expectedNeighbors);
		for(VertexData data : graph.getVertices()) {
			final int edgeCount = data.getEdges().size();
			if(edgeCount != targetNeighborCount) 
				throw new RuntimeException("The vertex "+data.getId()+" has "+edgeCount+" neighbors, expected "+targetNeighborCount);
		}

		final FeatureSpace space = graph.getFeatureSpace();
		for(VertexData data : graph.getVertices()) {
			for(Map.Entry<Integer, Float> entry : data.getEdges().entrySet()) {
				final float dist = space.computeDistance(data.getFeature(), graph.getVertexById(entry.getKey()).getFeature());
				if(entry.getValue() != dist)
					throw new RuntimeException("The vertex "+data.getId()+" has a weight "+entry.getValue()+" which is not equal to its distance "+dist);
			}
		}	

		return true;
	}


	// ----------------------------------------------------------------------------------------------
	// ----------------------------------- manipulation methods -------------------------------------
	// ----------------------------------------------------------------------------------------------

	@Override
	public void add(int label, FeatureVector data) {
		
		// check the feature vector compatibility
		if(data.dims() != graph.getFeatureSpace().dims() || data.getComponentType() != graph.getFeatureSpace().getComponentType())
			throw new RuntimeException("Invalid data component type "+data.getComponentType().getSimpleName()+" or dimension "+data.dims()+
									   ", expected "+graph.getFeatureSpace().getComponentType().getSimpleName()+" and "+graph.getFeatureSpace().dims());
		
		newEntryQueue.offer(new BuilderAddTask(label, manipulationCounter.getAndIncrement(), data));
	}

	@Override
	public void remove(int label) {
		removeEntryQueue.offer(new BuilderRemoveTask(label, manipulationCounter.getAndIncrement()));
	}

	@Override
	public void removeIf(IntPredicate labelFilter) {
		graph.getVertexLabels().forEach(label -> {
			if(labelFilter.test(label))
				remove(label);
		});
	}

	/**
	 * Add the new entry to the graph
	 * 
	 * @param addTask
	 */
	private void extendGraphByLabel(int newVertexLabel, FeatureVector newVertexFeature) {
		final FeatureSpace space = graph.getFeatureSpace();
		final int edgesPerVertex = graph.getEdgesPerVertex();

		if(graph.getVertexByLabel(newVertexLabel) != null)
			throw new RuntimeException("Graph contains vertex "+newVertexLabel+" already. Can not add it again.");

		// fully connect all vertices
		if(graph.getVertexCount() < graph.getEdgesPerVertex()+1) {			
			final VertexData newVertex = graph.addVertex(newVertexLabel, newVertexFeature);

			for(VertexData otherVertex : graph.getVertices()) {
				if(otherVertex.getId() != newVertex.getId()) {
					float weight = space.computeDistance(newVertexFeature, otherVertex.getFeature());
					graph.addUndirectedEdge(newVertex.getId(), otherVertex.getId(), weight);
				}
			}

			// extend the graph
		} else {

			// random entry point for the search
			VertexData entryVertex = null;
			final Iterator<VertexData> it = graph.getVertices().iterator();
			for (int i = 0; i < 1 + rnd.nextInt(graph.getVertexCount()); i++) 
				entryVertex = it.next();

			// find good neighbors for the new vertex
			final VertexDistance[] results = graph.search(Arrays.asList(newVertexFeature), extendK, extendEps, null, new int[] { entryVertex.getId() }).toArray(new VertexDistance[0]);

			// add an empty vertex to the graph (no neighbor information yet)
			final VertexData newVertex = graph.addVertex(newVertexLabel, newVertexFeature);

			// adding neighbors happens in two phases, the first tries to retain RNG, the second adds them without checking
			int checkRngPhase = 1;

			// remove an edge of the good neighbors and connect them with this new vertex
			final Map<Integer, Float> newNeighbors = newVertex.getEdges();
			while(newNeighbors.size() < edgesPerVertex) {
				for (int i = 0; i < results.length && newNeighbors.size() < edgesPerVertex; i++) {
					final VertexDistance candidate = results[i];
					final int candidateId = candidate.getVertexId();
					final float candidateWeight = candidate.getDistance();

					// check if the vertex is already in the edge list of the new vertex (added during a previous loop-run)
					// since all edges are undirected and the edge information of the new vertex does not yet exist, we search the other way around.
					if(newNeighbors.containsKey(candidateId)) 
						continue;

					// does the candidate has a neighbor which is connected to the new vertex and has a lower distance?
					if(checkRngPhase <= 1 && checkRNG(candidateId, newVertex.getId(), candidateWeight) == false) 
						continue;

					// This version is good for high LID data sets or small graphs with low distance count limit during ANNS
					int newNeighborIndex = 0;
					float newNeighborDistance = -1;
					{
						// find the worst edge of the new neighbor
						float newNeighborWeight = -1;						
						for(Map.Entry<Integer, Float> candiateNeighbor : graph.getVertexById(candidateId).getEdges().entrySet()) {
							final int candiateNeighborIndex = candiateNeighbor.getKey();

							// the suggested neighbor might already be in the edge list of the new vertex
							if(newNeighbors.containsKey(candiateNeighborIndex))
								continue;

							final float candiateNeighborWeight = candiateNeighbor.getValue();
							if(candiateNeighborWeight > newNeighborWeight) {
								newNeighborWeight = candiateNeighborWeight;
								newNeighborIndex = candiateNeighborIndex;
							}
						}

						if(newNeighborWeight == -1)
							continue;

						newNeighborDistance = space.computeDistance(newVertexFeature, graph.getVertexById(newNeighborIndex).getFeature()); 
					}

					// remove edge between the candidate and its worst neighbor
					graph.removeUndirectedEdge(candidateId, newNeighborIndex);

					// place the new vertex in the edge list of the result-vertex
					graph.addUndirectedEdge(newVertex.getId(), candidateId, candidateWeight);

					// place the new vertex in the edge list of the best edge neighbor
					graph.addUndirectedEdge(newVertex.getId(), newNeighborIndex, newNeighborDistance);

				}				
				checkRngPhase++;
			}

			if(newNeighbors.size() < edgesPerVertex) 
				throw new RuntimeException("Could find only "+newNeighbors.size()+" good neighbors for the new vertex "+newVertex.getId()+" need "+edgesPerVertex);

			// try to improve some of the non-perfect edges
			{
				// find all non-perfect edges
				final List<BoostedEdge> nonperfectNeighbors = new ArrayList<>();
				for (Map.Entry<Integer, Float> newNeighbor : newNeighbors.entrySet()) {
					final int neighborId = newNeighbor.getKey();

					// is this neighbor in the result list of the initial search request
					boolean isPerfect = false;
					for (int i = 0; i < results.length && isPerfect == false; i++) 
						if(results[i].getVertexId() == neighborId) 
							isPerfect = true;

					if(isPerfect == false) {
						final boolean rng = checkRNG(newVertex.getId(), neighborId, newNeighbor.getValue());
						nonperfectNeighbors.add(new BoostedEdge(neighborId, newNeighbor.getValue(), rng));
					}
				}

				// sort the non-perfect edges by their edge weight
				nonperfectNeighbors.sort((n1, n2) -> {
					int cmp = Float.compare(n1.weight, n2.weight);
					if(cmp == 0)
						cmp = Integer.compare(n1.vertexId, n2.vertexId);
					return cmp;
				});

				for (int i = 0; i < nonperfectNeighbors.size(); i++) {
					final BoostedEdge edge = nonperfectNeighbors.get(i);
					if(graph.hasEdge(newVertex.getId(), edge.vertexId) && (edge.rng == false || i < nonperfectNeighbors.size() / 2))
						improveEdges(newVertex.getId(), edge.vertexId, edge.weight);
				}
			}
		}
	}


	/**
	 * Try to improve the existing edge between the two vertices
	 * 
	 * @return true if a change could be made otherwise false
	 */
	private boolean improveEdges(final int vertex1, final int vertex2, final float dist12) {
		final List<BuilderChange> changes = new ArrayList<>();

		// remove the edge between vertex 1 and vertex 2
		graph.removeUndirectedEdge(vertex1, vertex2);
		changes.add(new BuilderChange(vertex1, vertex2, dist12, true));

		if(improveEdges(changes, vertex1, vertex2, vertex1, vertex1, dist12, 0) == false) {

			// undo all changes, in reverse order
			final int size = changes.size();
			for (int i = 0; i < size; i++) {
				final BuilderChange change = changes.get((size - 1) - i);
				if (change.removed == false) 
					graph.removeUndirectedEdge(change.id1, change.id2);
				else
					graph.addUndirectedEdge(change.id1, change.id2, change.weight);
			}
			return false;
		}		
		return true;
	}


	/**
	 * Do not call this method directly instead call improve() to improve the graph.
	 *  
	 * This is the extended part of the optimization process.
	 * The method takes an array where all graph changes will be documented.
	 * Vertex1 and vertex2 might be in a separate subgraph than vertex3 and vertex4.
	 * Thru a series of edges swaps both subgraphs should be reconnected..
	 * If those changes improve the graph this method returns true otherwise false. 
	 * 
	 * @return true if a good sequences of changes has been found
	 */
	private boolean improveEdges(final List<BuilderChange> changes, int vertex1, int vertex2, int vertex3, int vertex4, float totalGain, final int steps) {

		// 1. Find an edge for vertex2 which connects to the subgraph of vertex3 and vertex4. 
		//    Consider only vertices of the approximate nearest neighbor search. Since the 
		//    search started from vertex3 and vertex4 all vertices in the result list are in 
		//    their subgraph and would therefore connect the two potential subgraphs.	
		{
			final FeatureVector vertex2Feature = graph.getVertexById(vertex2).getFeature();
			final VertexDistance[] results = graph.search(Arrays.asList(vertex2Feature), improveK, improveEps, null, new int[] { vertex3, vertex4 }).toArray(new VertexDistance[0]);

			// find a good new vertex3
			float bestGain = totalGain;
			float dist23 = -1;
			float dist34 = -1;

			// We use the descending order to find the worst swap combination with the best gain
			// Sometimes the gain between the two best combinations is the same, its better to use one with the bad edges to make later improvements easier
			for(final VertexDistance result : results) {
				final int newVertex3 = result.getVertexId();

				// vertex1 and vertex2 got tested in the recursive call before and vertex4 got just disconnected from vertex2
				if(vertex1 != newVertex3 && vertex2 != newVertex3 && graph.hasEdge(vertex2, newVertex3) == false) {

					// 1.1 When vertex2 and the new vertex 3 gets connected, the full graph connectivity is assured again, 
					//     but the subgraph between vertex1/vertex2 and vertex3/vertex4 might just have one edge(vertex2, vertex3).
					//     Furthermore Vertex 3 has now to many edges, find an good edge to remove to improve the overall graph distortion. 
					//     FYI: If the just selected vertex3 is the same as the old vertex3, this process might cut its connection to vertex4 again.
					//     This will be fixed in the next step or until the recursion reaches max_path_length.
					for(Map.Entry<Integer, Float> edge : graph.getVertexById(newVertex3).getEdges().entrySet()) {
						final int newVertex4 = edge.getKey();
						final float newVertex4Weight = edge.getValue();

						// compute the gain of the graph distortion if this change would be applied
						final float gain = (totalGain - result.getDistance()) + newVertex4Weight;

						// do not remove the edge which was just added
						if(newVertex4 != vertex2 && bestGain < gain) {
							bestGain = gain;
							vertex3 = newVertex3;
							vertex4 = newVertex4;
							dist23 = result.getDistance();
							dist34 = newVertex4Weight;    
						}
					}
				}
			}

			// no new vertex3 was found
			if(dist23 == -1)
				return false;

			// connect vertex2 with vertex3. 
			totalGain -= dist23;
			graph.addUndirectedEdge(vertex2, vertex3, dist23);
			changes.add(new BuilderChange(vertex2, vertex3, dist23, false));

			// remove edge between vertex3 and vertex4
			totalGain += dist34;
			graph.removeUndirectedEdge(vertex3, vertex4);
			changes.add(new BuilderChange(vertex3, vertex4, dist34, true));
		}

		// 2. Try to connect vertex1 with vertex4
		{
			final FeatureSpace space = graph.getFeatureSpace();

			// 2.1a Vertex1 and vertex4 might be the same. This is quite the rare case, but would mean there are two edges missing.
			//     Proceed like extending the graph:
			//     Search for a good vertex to connect to, remove its worst edge and connect
			//     both vertices of the worst edge to the vertex4. Skip the edge any of the two
			//     two vertices are already connected to vertex4.
			if(vertex1 == vertex4) {

				// find a good (not yet connected) vertex for vertex1/vertex4
				final FeatureVector vertex4Feature = graph.getVertexById(vertex4).getFeature();
				final VertexDistance[] results = graph.search(Arrays.asList(vertex4Feature), improveK, improveEps, null, new int[] { vertex2, vertex3 }).toArray(new VertexDistance[0]);

				float bestGain = 0;
				int bestSelectedNeighbor = 0;
				float bestSelectedNeighborDist = 0;
				float bestSelectedNeighborOldDist = 0;
				int bestGoodVertex = 0;
				float bestGoodVertexDist = 0;
				for(VertexDistance result : results) {
					final int goodVertex = result.getVertexId();

					// the new vertex should not be connected to vertex4 yet
					if(vertex4 != goodVertex && graph.hasEdge(vertex4, goodVertex) == false) {
						final float goodVertexDist = result.getDistance();

						// select any edge of the good vertex which improves the graph quality when replaced with a connection to vertex 4
						for(Map.Entry<Integer, Float> edge : graph.getVertexById(goodVertex).getEdges().entrySet()) {
							final int selectedNeighbor = edge.getKey();

							// ignore edges where the second vertex is already connect to vertex4
							if(vertex4 != selectedNeighbor && graph.hasEdge(vertex4, selectedNeighbor) == false) {			                
								final float oldNeighborDist = edge.getValue();
								final float newNeighborDist = space.computeDistance(vertex4Feature, graph.getVertexById(selectedNeighbor).getFeature());

								// do all the changes improve the graph?
								float newGain = (totalGain + oldNeighborDist) - (goodVertexDist + newNeighborDist);
								if(bestGain < newGain) {
									bestGain = newGain;
									bestSelectedNeighbor = selectedNeighbor;
									bestSelectedNeighborDist = newNeighborDist;
									bestSelectedNeighborOldDist = oldNeighborDist;
									bestGoodVertex = goodVertex;
									bestGoodVertexDist = goodVertexDist;
								}
							}
						}
					}
				}

				if(bestGain > 0)
				{
					// remove edge between the good vertex and one of its neighbors
					graph.removeUndirectedEdge(bestGoodVertex, bestSelectedNeighbor);
					changes.add(new BuilderChange(bestGoodVertex, bestSelectedNeighbor, bestSelectedNeighborOldDist, true));
								
					// connect vertex4/vertex1 with the good vertex and its selected neighbor
					graph.addUndirectedEdge(vertex1, bestGoodVertex, bestGoodVertexDist);
					changes.add(new BuilderChange(vertex1, bestGoodVertex, bestGoodVertexDist, false));
					graph.addUndirectedEdge(vertex1, bestSelectedNeighbor, bestSelectedNeighborDist);
					changes.add(new BuilderChange(vertex1, bestSelectedNeighbor, bestSelectedNeighborDist, false));

					return true;
				}

			} else {

				// 2.1b If there is a way from vertex2 or vertex3, to vertex1 or vertex4 then ...
				//      Try to connect vertex1 with vertex4
				//      Much more likely than 2.1a 
				if(graph.hasEdge(vertex1, vertex4) == false) {

					// Is the total of all changes still beneficial?
					final float dist14 = space.computeDistance(graph.getVertexById(vertex1).getFeature(), graph.getVertexById(vertex4).getFeature());
					if((totalGain - dist14) > 0) {

						final int[] entryVertices = { vertex2, vertex3 }; 
						if(graph.hasPath(entryVertices, vertex1, improveK, improveEps).size() > 0 || graph.hasPath(entryVertices, vertex4, improveK, improveEps).size() > 0) {

							// connect vertex1 with vertex4
							graph.addUndirectedEdge(vertex1, vertex4, dist14);
							changes.add(new BuilderChange(vertex1, vertex4, dist14, false));

							return true;
						}
					}
				}
			}
		}
		
		// 3. Maximum path length
		if(steps >= maxPathLength) 
			return false;

		// 4. swap vertex1 and vertex4 every second round, to give each a fair chance
		if(steps % 2 == 1) {
			final int b = vertex1;
			vertex1 = vertex4;
			vertex4 = b;
		}

		// 5. early stop
		if(totalGain < 0) 
			return false;

		return improveEdges(changes, vertex1, vertex4, vertex2, vertex3, totalGain, steps + 1);
	}



	/**
	 * Is the vertexId a RNG conform neighbor if it gets connected to targetId?
	 * 
	 * Does vertexId has a neighbor which is connected to the targetId and has a lower weight?
	 * 
	 * @param vertexId
	 * @param targetId
	 * @param vertexTargetWeight
	 * @return
	 */
	private boolean checkRNG(int vertexId, int targetId, float vertexTargetWeight) {
		for(Map.Entry<Integer, Float> neighbor : graph.getVertexById(vertexId).getEdges().entrySet()) {
			float neighborTargetWeight = graph.getEdgeWeight(neighbor.getKey(), targetId);
			if(neighborTargetWeight >= 0 && vertexTargetWeight > Math.max(neighbor.getValue(), neighborTargetWeight)) 
				return false;
		}
		return true;
	}


	/**
	 * Remove the entry from the graph
	 *  
	 * @param removeTask
	 */
	private void shrinkGraphByLabel(int label) {
		final int edgesPerVertex = graph.getEdgesPerVertex();
		final FeatureSpace space = graph.getFeatureSpace();
		final List<BuilderChange> changes = new ArrayList<>();

		// 1 remove the vertex and collect the vertices which are missing an edge
		final Set<Integer> involvedVertices = graph.removeVertexByLabel(label).keySet();
		
		// 1.1 handle the use case where the graph does not have enough vertices to fulfill the edgesPerVertex requirement
		//     and just remove the vertex without reconnecting the involved vertices because they are all fully connected
		if(graph.getVertexCount() <= edgesPerVertex)
			return;

		// 2 find pairs or groups of vertices which can reach each other
		final Map<Integer, Set<Integer>> reachability = new HashMap<>();
		{
			
			// 2.1 start with checking the adjacent neighbors
			for (int involvedVertex : involvedVertices) {
				final Set<Integer> reachableVertices = reachability.computeIfAbsent(involvedVertex, k -> new HashSet<>(Arrays.asList(k)));
				
				// is any of the adjacent neighbors of involvedVertex also in the set of involvedVertices
				for(int neighborId : graph.getVertexById(involvedVertex).getEdges().keySet()) {
					if(involvedVertices.contains(neighborId) && reachableVertices.contains(neighborId) == false) {
						
						// if this neighbor does not have a set of reachable vertices yet, share the current set reachableVertices
						final Set<Integer> neighborsReachableVertices = reachability.get(neighborId);
						if(neighborsReachableVertices == null) { 
							reachableVertices.add(neighborId);
							reachability.put(neighborId, reachableVertices);
						} else {
							
							// if the neighbor already has a set of reachable vertices, copy them over and replace all their references to the new and bigger set
							reachableVertices.addAll(neighborsReachableVertices);
							for (int neighborsReachableVertex : neighborsReachableVertices) 
								reachability.put(neighborsReachableVertex, reachableVertices);
						}
					}
				}
			}
			
			// 2.2 use graph.hasPath(...) to find a path for every not paired but involved vertex, to any other involved vertex 
			for (Map.Entry<Integer, Set<Integer>> vertexReachability : reachability.entrySet()) {
				final int involvedVertex = vertexReachability.getKey();
				final Set<Integer> reachableVertices = vertexReachability.getValue();
				
				// during 2.1 each vertex got a set of reachable vertices with at least one entry (the vertex itself)
				// all vertices containing only one element still need to find a other reachable vertex 
				if(reachableVertices.size() <= 1) {
					
					// is there a path from any of the other involvedVertices to the lonely vertex?
					final int[] fromVertices = involvedVertices.stream().mapToInt(Integer::intValue).filter(v -> v != involvedVertex).toArray();
					List<VertexDistance> traceback = graph.hasPath(fromVertices, involvedVertex, improveK, improveEps);
					if(traceback.size() == 0) {
						// TODO implement flood full to find an involved vertex without compute distances
						traceback = graph.hasPath(fromVertices, involvedVertex, graph.getVertexCount(), 1);
					}
					
					// the first vertex in the traceback path must be one of the other involved vertices
					final int reachableVertex = traceback.get(traceback.size()-1).getVertexId();
					final Set<Integer> reachableVerticesOfReachableVertex = reachability.get(reachableVertex);

					// add the involvedVertex to its reachable set and replace the reachable set of the involvedVertex 
					reachableVerticesOfReachableVertex.add(involvedVertex);
					vertexReachability.setValue(reachableVerticesOfReachableVertex);
				}
			}
		}
		
		// 3 reconnect the groups
		{
			final Function<Collection<Integer>, VertexData[]> idsToVertexArray = ids -> ids.stream().map(id -> graph.getVertexById(id)).toArray(VertexData[]::new);
			final VertexData[][] reachableGroups = reachability.values().stream().distinct().map(idsToVertexArray).toArray(VertexData[][]::new);
						
			// 3.1 find the biggest group and connect each of its vertices to one of the smaller groups
			//      Stop when all groups are connected or every vertex in the big group got an additional edge.
			//      In case of the later, repeat the process with the next biggest group.
			if(reachableGroups.length > 1) {
				
				// sort the groups by size 			
				Arrays.sort(reachableGroups, Comparator.comparingInt((VertexData[] g) -> g.length).reversed());
				
				// find the next biggest group
				for (int g = 0, n = 1; g < reachableGroups.length && n < reachableGroups.length; g++) {
					final VertexData[] reachableGroup = reachableGroups[g];
					
					// iterate over all its entries to find a vertex which is still missing an edge
					nextVertex: for (int i = 0; i < reachableGroup.length && n < reachableGroups.length; i++) {
						final VertexData vertex = reachableGroup[i];
						if(vertex.getEdges().size() < edgesPerVertex) {
							
							// find another vertex in a smaller group, also missing an edge			
							for (; n < reachableGroups.length; n++) {								
								final VertexData[] otherGroup = reachableGroups[n];
								for (int j = 0; j < otherGroup.length; j++) {
									final VertexData otherVertex = otherGroup[j];
									if(otherVertex.getEdges().size() < edgesPerVertex) {
										
										// connect vertexId and otherVertexId
										final float weight = space.computeDistance(vertex.getFeature(), otherVertex.getFeature());
										graph.addUndirectedEdge(vertex.getId(), otherVertex.getId(), weight);
										changes.add(new BuilderChange(vertex.getId(), otherVertex.getId(), weight, false));
										
										// repeat until all small groups are connected
										n++;
										continue nextVertex;
									}
								}
							}
						}
					}
				}
			}
			
			// 3.1 now all groups are reachable but still some vertices are missing edge, try to connect them to each other.
			final VertexData[] remainingVertices = Stream.of(reachableGroups).flatMap(Stream::of).filter(v -> v.getEdges().size() < edgesPerVertex).toArray(VertexData[]::new);
			for (int i = 0; i < remainingVertices.length; i++) {
				final VertexData vertexA = remainingVertices[i];
				if(vertexA.getEdges().size() < edgesPerVertex) {
					
					// find a vertexB with the smallest distance to A
					int bestVertexBId = -1;
					float bestDistance = Float.MAX_VALUE;
					for (int j = i+1; j < remainingVertices.length; j++) {
						final VertexData vertexB = remainingVertices[j];
						if(vertexB.getEdges().size() < edgesPerVertex && vertexA.getEdges().containsKey(vertexB.getId()) == false) {
							
							final float weight = space.computeDistance(vertexA.getFeature(), vertexB.getFeature());
							if(weight < bestDistance) {
								bestDistance = weight;
								bestVertexBId = vertexB.getId();
							}
						}
					}
					
					// connect vertexA and vertexB
					if(bestVertexBId >= 0) {
						graph.addUndirectedEdge(vertexA.getId(), bestVertexBId, bestDistance);
						changes.add(new BuilderChange(vertexA.getId(), bestVertexBId, bestDistance, false));
					}
				}
			}

				
			// 3.2 the remaining vertices can not be connected to any of the other involved vertices, because they already have an edge to all of them.
			for (int i = 0; i < remainingVertices.length; i++) {
				final VertexData vertexA = remainingVertices[i];
				if(vertexA.getEdges().size() < edgesPerVertex) {
					
					// scan the neighbors of the adjacent vertices of A and find a vertex B with the smallest distance to A
					VertexData vertexB = null; 
					float weightAB = Float.MAX_VALUE;
					for (int neighborOfA : vertexA.getEdges().keySet()) {
						for (int potentialVertexBId : graph.getVertexById(neighborOfA).getEdges().keySet()) {
							if(vertexA.getId() != potentialVertexBId && vertexA.getEdges().containsKey(potentialVertexBId) == false) {	
								final VertexData potentialVertexB = graph.getVertexById(potentialVertexBId);
								final float weight = space.computeDistance(vertexA.getFeature(), potentialVertexB.getFeature());
								if(weight < weightAB) {
									weightAB = weight;
									vertexB = potentialVertexB;
								}
							}
						}
					}

					// Get another vertex missing an edge called C and at this point sharing an edge with A (by definition of 3.2)
					for (int j = i+1; j < remainingVertices.length; j++) {
						final VertexData vertexC = remainingVertices[j];
						if(vertexC.getEdges().size() < edgesPerVertex) {
														
							// check the neighborhood of B to find a vertex D not yet adjacent to C but with the smallest possible distance to D
							VertexData vertexD = null; 
							float weightCD = Float.MAX_VALUE;
							for (int potentialVertexDId : vertexB.getEdges().keySet()) {
								if(vertexA.getId() != potentialVertexDId && vertexC.getId() != potentialVertexDId && vertexC.getEdges().containsKey(potentialVertexDId) == false) {
									final VertexData potentialVertexD = graph.getVertexById(potentialVertexDId);
									final float weight = space.computeDistance(vertexC.getFeature(), potentialVertexD.getFeature());
									if(weight < weightCD) {
										weightCD = weight;
										vertexD = potentialVertexD;
									}
								}
							}
							
	
							// replace edge between B and D, with one between A and B as well as C and D
							changes.add(new BuilderChange(vertexB.getId(), vertexD.getId(), graph.getEdgeWeight(vertexB.getId(), vertexD.getId()), true));
							graph.removeUndirectedEdge(vertexB.getId(), vertexD.getId());
	
							changes.add(new BuilderChange(vertexA.getId(), vertexB.getId(), weightAB, false));
							graph.addUndirectedEdge(vertexA.getId(), vertexB.getId(), weightAB);
	
							changes.add(new BuilderChange(vertexC.getId(), vertexD.getId(), weightCD, false));
							graph.addUndirectedEdge(vertexC.getId(), vertexD.getId(), weightCD);
							
							break;
						}
					}
				}
			}
		}
	}


	@Override
	public synchronized void build(ChangeListener listener) {
		stopBuilding = false;

		long step = 0;
		long added = 0;
		long deleted = 0;
		long improved = 0;
		long tries = 0;
		int lastAdd = -1; 
		int lastDelete = -1;

		while(stopBuilding == false) {

			// add or delete a vertex
			if(this.newEntryQueue.size() > 0 || this.removeEntryQueue.size() > 0) {
				long addTaskManipulationIndex = Long.MAX_VALUE;
				long delTaskManipulationIndex = Long.MAX_VALUE;

				if(this.newEntryQueue.size() > 0) 
					addTaskManipulationIndex = this.newEntryQueue.peek().manipulationIndex;

				if(this.removeEntryQueue.size() > 0) 
					delTaskManipulationIndex = this.removeEntryQueue.peek().manipulationIndex;

				if(addTaskManipulationIndex < delTaskManipulationIndex) {
					BuilderAddTask addTask = this.newEntryQueue.poll();
					extendGraphByLabel(addTask.label, addTask.feature);
					added++;
					lastAdd = addTask.label;
				} else {
					BuilderRemoveTask removeTask = this.removeEntryQueue.poll();
					shrinkGraphByLabel(removeTask.label);
					deleted++;
					lastDelete = removeTask.label;
				}
			}

			//try to improve the graph
			//	        if(graph.getVertexCount() > graph.getEdgesPerVertex() && improve_k_ > 0) {
			//	          for (int64_t swap_try = 0; swap_try < int64_t(this->swap_tries_); swap_try++) {
			//	            tries++;
			//
			//	            if(this->improveEdges()) {
			//	              improved++;
			//	              swap_try -= this->additional_swap_tries_;
			//	            }
			//	          }
			//	        }

			step++;

			// inform the listener
			listener.onChange(step, added, deleted, improved, tries, lastAdd, lastDelete);
			lastAdd = -1; 
			lastDelete = -1;
		}

		stopBuilding = true;
	}

	@Override
	public void stop() {
		stopBuilding = true;
	}


	/**
	 * 
	 * @author Nico Hezel
	 */
	protected static class BuilderChange {

		protected final int id1;
		protected final int id2;
		protected final float weight;
		protected final boolean removed;

		public BuilderChange(int id1, int id2, float weight, boolean removed) {
			this.id1 = id1;
			this.id2 = id2;
			this.weight = weight;
			this.removed = removed;
		} 

		@Override
		public String toString() { 
			return ((removed) ? "Removed" : "Added") + " edge between "+id1+" and "+id2+" with weight "+weight;
		}
	}

	/**
	 * 
	 * @author Nico Hezel
	 */
	protected static class BoostedEdge {

		protected final int vertexId;
		protected final float weight;
		protected final boolean rng;

		public BoostedEdge(int vertexId, float weight, boolean rng) {
			this.vertexId = vertexId;
			this.weight = weight;
			this.rng = rng;
		}
	}

	/**
	 * 
	 * @author Nico Hezel
	 */
	public static class BuilderAddTask {
		public final int label;
		public final long manipulationIndex;
		public final FeatureVector feature;
		public BuilderAddTask(int label, long manipulationIndex, FeatureVector feature) {
			super();
			this.label = label;
			this.manipulationIndex = manipulationIndex;
			this.feature = feature;
		}
	}

	/**
	 * 
	 * @author Nico Hezel
	 */
	public static class BuilderRemoveTask {
		public final int label;
		public final long manipulationIndex;
		public BuilderRemoveTask(int label, long manipulationIndex) {
			super();
			this.label = label;
			this.manipulationIndex = manipulationIndex;
		}
	}
}
