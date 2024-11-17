package com.vc.deg.ref.designer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.stream.Collectors;

import com.koloboke.collect.map.IntFloatCursor;
import com.koloboke.collect.map.IntIntMap;
import com.koloboke.collect.map.hash.HashIntIntMaps;
import com.koloboke.collect.set.IntSet;
import com.koloboke.collect.set.ObjSet;
import com.koloboke.collect.set.hash.HashIntSets;
import com.koloboke.collect.set.hash.HashObjSets;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.graph.GraphDesigner;
import com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph;
import com.vc.deg.ref.graph.QueryDistance;
import com.vc.deg.ref.graph.VertexData;


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

	protected boolean schemaC;

	public EvenRegularGraphDesigner(ArrayBasedWeightedUndirectedRegularGraph graph) {
		this.graph = graph;
		this.rnd = new Random(7);
		this.manipulationCounter = new AtomicLong(0);
		this.newEntryQueue = new ConcurrentLinkedQueue<>();
		this.removeEntryQueue = new ConcurrentLinkedQueue<>();

		// default hyper parameters
		setExtendK(graph.getEdgesPerVertex() * 2);
		setExtendEps(0.2f);
		setExtendSchema(true);
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
	public void setExtendSchema(boolean useSchemaC) {
		schemaC = useSchemaC;
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
			final IntFloatCursor edgeCursor = data.getEdges().cursor();
			while(edgeCursor.moveNext()) {
				if(edgeCursor.key() != data.getId()) {
					sum += edgeCursor.value();
					count++;
				}
			}
		}
		return (float)(sum/count);
	}

	@Override
	public float calcAvgNeighborRank() {
		return calcAvgNeighborRank(null);
	}

	@Override
	public float calcAvgNeighborRank(int[][] topLists) {

		// array of all vertex ids
		final int[] testOrder = graph.getVertices().stream().mapToInt(VertexData::getId).toArray();

		// shuffle
		final Random rand = new Random(7);
		for (int i = 0; i < testOrder.length; i++) {
			int randomIndexToSwap = rand.nextInt(testOrder.length);
			int temp = testOrder[randomIndexToSwap];
			testOrder[randomIndexToSwap] = testOrder[i];
			testOrder[i] = temp;
		}


		final AtomicInteger queryCount = new AtomicInteger(0);
		final AtomicReference<Double> rankSum = new AtomicReference<>(Double.valueOf(0));

		Arrays.stream(testOrder).parallel().forEach(queryId -> {
			double avgNeighborRank;
			if(topLists == null)
				avgNeighborRank = calcAvgNeighborRank(queryId);
			else {
				avgNeighborRank = calcAvgNeighborRank(queryId, topLists[queryId]);
				if(avgNeighborRank == -1)
					avgNeighborRank = calcAvgNeighborRank(queryId);
			}

			final double currentRankSum = rankSum.accumulateAndGet(avgNeighborRank, (x, y) -> (x + y));
			final int currentQueryCount = queryCount.incrementAndGet();
			if(currentQueryCount % 100 == 0) 
				System.out.println("AvgNeighborRank after "+currentQueryCount+" vertices is "+(currentRankSum/currentQueryCount));
		});
		//		for (int queryId : testOrder) {
		//			
		//			double avgNeighborRank = calcAvgNeighborRank(queryId, topLists[queryId]);
		//			if(avgNeighborRank == -1)
		//				avgNeighborRank = calcAvgNeighborRank(queryId);
		//			
		//			rankSum.accumulateAndGet(avgNeighborRank, (x, y) -> (x + y));
		//			if(queryCount.incrementAndGet() % 100 == 0) 
		//				System.out.println("AvgNeighborRank after "+queryCount+" vertices is "+(rankSum.get()/queryCount.get()));
		//		}
		return (float)(rankSum.get()/queryCount.get());
	}

	/**
	 * Compute the average neighbor rank of the vertex with the help of the top list.
	 * If the top list is to short return -1;
	 * 
	 * @param queryId
	 * @param topList
	 * @return
	 */
	protected double calcAvgNeighborRank(int queryId, int[] topList) {
		final VertexData query = graph.getVertexById(queryId);

		// compute ranks of neighbors
		long neighborCount = 0;
		double neighborRankSum = 0;
		for(int neighborId : query.getEdges().keySet()) {
			if(neighborId != queryId) {
				int rank = 0;
				for (; rank < topList.length; rank++) {
					if(neighborId == topList[rank]) 
						break;
				}

				if(rank == topList.length)
					return -1;

				neighborRankSum += rank + 1; // rank 0 is always a self reference which is not included in the top list
				neighborCount++;
			}
		}

		return neighborRankSum / neighborCount;
	}


	/**
	 * Compute the average neighbor rank of the vertex
	 * 
	 * @param queryId
	 * @return
	 */
	protected double calcAvgNeighborRank(int queryId) {

		class IntFloat {
			int index;
			float distance;
			public IntFloat(int index, float distance) {
				this.index = index;
				this.distance = distance;
			}
		}

		final int size = graph.getVertices().size();
		final VertexData query = graph.getVertexById(queryId);
		final FeatureVector queryFeature = query.getFeature();

		// compute sorted rank list
		final FeatureSpace space = graph.getFeatureSpace();
		final List<IntFloat> idDistanceList = new ArrayList<>(size);
		for(VertexData data : graph.getVertices()) {
			final float dist = space.computeDistance(queryFeature, data.getFeature());
			idDistanceList.add(new IntFloat(data.getId(), dist));
		}
		idDistanceList.sort((o1, o2) -> Float.compare(o1.distance, o2.distance));

		// compute ranks of neighbors
		long neighborCount = 0;
		double neighborRankSum = 0;
		for(int neighborId : query.getEdges().keySet()) {
			if(neighborId != queryId) {
				int rank = 0;
				for (; rank < size; rank++) {
					if(neighborId == idDistanceList.get(rank).index)
						break;
				}
				neighborRankSum += rank;
				neighborCount++;
			}
		}

		return neighborRankSum / neighborCount;
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
				if (Math.abs(entry.getValue() - dist) > 0.001f)
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
			final VertexData entryVertex = graph.getRandomVertex(rnd);

			// find good neighbors for the new vertex
			final QueryDistance[] results = graph.search(Arrays.asList(newVertexFeature), extendK, extendEps, null, new int[] { entryVertex.getId() }, true).toArray(new QueryDistance[0]);

			// add an empty vertex to the graph (no neighbor information yet)
			final VertexData newVertex = graph.addVertex(newVertexLabel, newVertexFeature);

			// adding neighbors happens in two phases, the first tries to retain RNG, the second adds them without checking
			int checkRngPhase = 1;

			// remove an edge of the good neighbors and connect them with this new vertex
			final Map<Integer, Float> newNeighbors = newVertex.getEdges();
			while(newNeighbors.size() < edgesPerVertex) {
				for (int i = 0; i < results.length && newNeighbors.size() < edgesPerVertex; i++) {
					final QueryDistance candidate = results[i];
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
					int newNeighborId = 0;
					float newNeighborDistance = -1;
					if(schemaC) {
						// find the worst edge of the new neighbor
						float newNeighborWeight = -1;						
						for(Map.Entry<Integer, Float> candiateNeighbor : graph.getVertexById(candidateId).getEdges().entrySet()) {
							final int candiateNeighborId = candiateNeighbor.getKey();

							// the suggested neighbor might already be in the edge list of the new vertex
							if(newNeighbors.containsKey(candiateNeighborId))
								continue;

							// find the worst neighbor
							final float candiateNeighborWeight = candiateNeighbor.getValue();
							if(candiateNeighborWeight > newNeighborWeight) {
								newNeighborWeight = candiateNeighborWeight;
								newNeighborId = candiateNeighborId;
							}
						}

						if(newNeighborWeight == -1)
							continue;

						newNeighborDistance = space.computeDistance(newVertexFeature, graph.getVertexById(newNeighborId).getFeature()); 
					}
					else
					{
						// find the edge which improves the distortion the most: (distance_new_edge1 + distance_new_edge2) - distance_removed_edge  
						float bestDistortion = Float.MAX_VALUE;
						for(Map.Entry<Integer, Float> candiateNeighbor : graph.getVertexById(candidateId).getEdges().entrySet()) {
							final int candiateNeighborId = candiateNeighbor.getKey();

							// the suggested neighbor might already be in the edge list of the new vertex
							if(newNeighbors.containsKey(candiateNeighborId))
								continue;

							// take the neighbor with the best distance to the new vertex
							final float neighborDistance = space.computeDistance(newVertexFeature, graph.getVertexById(candiateNeighborId).getFeature());
							float distortion = (candidateWeight + neighborDistance) - candiateNeighbor.getValue();
							if(distortion < bestDistortion) {
								bestDistortion = distortion;
								newNeighborId = candiateNeighborId;
								newNeighborDistance = neighborDistance;
							} 
						}
					}

					// remove edge between the candidate and its worst neighbor
					graph.removeUndirectedEdge(candidateId, newNeighborId);

					// place the new vertex in the edge list of the result-vertex
					graph.addUndirectedEdge(newVertex.getId(), candidateId, candidateWeight);

					// place the new vertex in the edge list of the best edge neighbor
					graph.addUndirectedEdge(newVertex.getId(), newNeighborId, newNeighborDistance);

				}				
				checkRngPhase++;
			}

			if(newNeighbors.size() < edgesPerVertex) 
				throw new RuntimeException("Could find only "+newNeighbors.size()+" good neighbors for the new vertex "+newVertex.getId()+" need "+edgesPerVertex);
		}
	}

	/**
	 * Remove an random edge
	 * 
	 * @return
	 */
	private boolean improveEdges() {
		// TODO add code
		return false;
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
			final QueryDistance[] results = graph.search(Arrays.asList(vertex2Feature), improveK, improveEps, null, new int[] { vertex3, vertex4 }, true).toArray(new QueryDistance[0]);

			// find a good new vertex3
			float bestGain = totalGain;
			float dist23 = -1;
			float dist34 = -1;

			// We use the descending order to find the worst swap combination with the best gain
			// Sometimes the gain between the two best combinations is the same, its better to use one with the bad edges to make later improvements easier
			for(final QueryDistance result : results) {
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
				final QueryDistance[] results = graph.search(Arrays.asList(vertex4Feature), improveK, improveEps, null, new int[] { vertex2, vertex3 }, true).toArray(new QueryDistance[0]);

				float bestGain = 0;
				int bestSelectedNeighbor = 0;
				float bestSelectedNeighborDist = 0;
				float bestSelectedNeighborOldDist = 0;
				int bestGoodVertex = 0;
				float bestGoodVertexDist = 0;
				for(QueryDistance result : results) {
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
	 * @param vertexId
	 * @param targetId
	 * @param vertexTargetWeight
	 * @return
	 */
	private boolean checkRNG(int vertexId, int targetId, float vertexTargetWeight) {
		for(Map.Entry<Integer, Float> neighbor : graph.getVertexById(vertexId).getEdges().entrySet()) {
			float neighborTargetWeight = graph.getEdgeWeight(neighbor.getKey(), targetId);

			// Does vertexId has a neighbor which is connected to the targetId and has a lower weight?
			if(neighborTargetWeight >= 0 && vertexTargetWeight > Math.max(neighbor.getValue(), neighborTargetWeight)) 
				return false;
		}
		return true;
	}


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
		final ObjSet<ReachableGroup> uniqueGroups = HashObjSets.newMutableSet();		
		{
			{
				final UnionFind pathMap = new UnionFind(edgesPerVertex); // TODO only use for neighbor check
				final Map<Integer, ReachableGroup> reachableGroups = new HashMap<>(edgesPerVertex);
				involvedVertices.forEach(involvedVertex -> {
					reachableGroups.computeIfAbsent(involvedVertex, k -> new ReachableGroup(k));
					pathMap.update(involvedVertex, involvedVertex);
				});

				final Function<UnionFind, Boolean> enoughFreeConections = (parents) -> {

					int isolatedCount = 0;
					int avaiableConnectionCount = 0;	
					for (int involvedVertex : involvedVertices) {
						final int reachableVertex = parents.find(involvedVertex);
						if(involvedVertex == reachableVertex) {
							final ReachableGroup group = reachableGroups.get(reachableVertex);
							if(group.size() == 1)
								isolatedCount++;
							else if(group.getMissingEdges().size() > 2)
								avaiableConnectionCount += group.getMissingEdges().size() - 2;
						}
					}
					return avaiableConnectionCount < isolatedCount;
				};	

				//	        	long startNeighborCheck = System.currentTimeMillis();

				// 2.1 start with checking the adjacent neighbors
				int neighborCheckDepth = 0;
				IntSet check = HashIntSets.newMutableSet(involvedVertices);
				IntSet checkNext = HashIntSets.newMutableSet();				
				while(enoughFreeConections.apply(pathMap)) {
					for (int checkVertex : check) {
						final int involvedVertex = pathMap.find(checkVertex);
						final ReachableGroup reachableGroup = reachableGroups.get(involvedVertex);

						// check only involved vertices and vertices which can only reach 1 involved vertex
						// no need for big groups to find other groups at the expense of processing power
						if(neighborCheckDepth > 0 && reachableGroup.size() > 1)
							continue;

						// check the neighbors of checkVertex if they can reach another reachableGroup
						for(int neighbor : graph.getVertexById(checkVertex).getEdges().keySet()) {
							final int otherInvolvedVertex = pathMap.find(neighbor);

							// neighbor is not yet in the union find
							if(otherInvolvedVertex == -1) {

								pathMap.update(neighbor, involvedVertex);
								checkNext.add(neighbor);
							}
							// the neighbor can reach another involved vertex
							else if(otherInvolvedVertex != involvedVertex) {							
								pathMap.update(otherInvolvedVertex, involvedVertex);
								reachableGroup.merge(reachableGroups.get(otherInvolvedVertex));
							}
						}
					}

					// swap
					IntSet buff = check;
					check = checkNext;
					checkNext = buff;
					checkNext.clear();

					neighborCheckDepth++;
				}

				for(int involvedVertex : involvedVertices) {
					final ReachableGroup group = reachableGroups.get(pathMap.find(involvedVertex));
					uniqueGroups.add(group);
				}	
			}

			// 2.2 get all isolated vertices
			final List<ReachableGroup> isolatedGroups = new ArrayList<>(edgesPerVertex);
			for (ReachableGroup vertexReach : uniqueGroups) 
				if(vertexReach.size() == 1)
					isolatedGroups.add(vertexReach);

			// 2.3 find for every isolated vertex the best other involved vertex which is part of a reachable group
			for (ReachableGroup isolatedVertexGroup : isolatedGroups) {

				// are you still isolated?
				if(isolatedVertexGroup.size() > 1) 
					continue;

				final int isolatedVertex = isolatedVertexGroup.getVertexIndex();
				final FeatureVector isolatedVertexFeature = graph.getVertexById(isolatedVertex).getFeature();

				// check the reachable groups for good candidates which can connect to the isolated vertex
				float bestCandidateDistance = Float.MAX_VALUE;
				ReachableGroup bestCandidateGroup = null;
				int bestCandidateIndex = -1;
				for (ReachableGroup candidateGroup : uniqueGroups) {

					// skip all groups which do not have enough vertices missing an edge
					final Set<Integer> missingEdges = candidateGroup.getMissingEdges();
					if(missingEdges.size() <= 2)
						continue;

					// find the candidate with the best distance to the isolated vertex
					for (int candidate : missingEdges) {
						//						if(graph.hasEdge(isolatedVertex, candidate) == false) {
						final FeatureVector candidateFeature = graph.getVertexById(candidate).getFeature();
						final float distance = space.computeDistance(candidateFeature, isolatedVertexFeature);
						if(distance < bestCandidateDistance) {
							bestCandidateDistance = distance;
							bestCandidateIndex = candidate;
							bestCandidateGroup = candidateGroup;
						}
						//						}
					}
				}

				// found a good candidate, add the isolated vertex to its reachable group and an edge between them
				graph.addUndirectedEdge(isolatedVertex, bestCandidateIndex, bestCandidateDistance);
				changes.add(new BuilderChange(isolatedVertex, bestCandidateIndex, bestCandidateDistance, false));

				// merge groups
				bestCandidateGroup.hasEdge(bestCandidateIndex);
				isolatedVertexGroup.hasEdge(isolatedVertex);
				bestCandidateGroup.merge(isolatedVertexGroup);

				uniqueGroups.remove(isolatedVertexGroup);
			}
		}

		// 3 reconnect the groups
		{
			// Sort the unique groups by size in ascending order
			final List<ReachableGroup> reachableGroups = new ArrayList<>(uniqueGroups);
			Collections.sort(reachableGroups, Comparator.comparingInt((ReachableGroup g) -> g.size()).reversed());

			// 3.1 Find the biggest group and one of its vertices to one vertex of a smaller group. Repeat until only one group is left.
			while(reachableGroups.size() >= 2) {
				final ReachableGroup reachableGroup = reachableGroups.get(reachableGroups.size()-1);
				final ReachableGroup otherGroup = reachableGroups.get(reachableGroups.size()-2);
				final Set<Integer> reachableVertices = reachableGroup.getMissingEdges();
				final Set<Integer> otherVertices = otherGroup.getMissingEdges();

				float bestDistance = Float.MAX_VALUE;
				int bestOtherVertex = -1;
				int bestReachableVertex = -1;

				// find another vertex in a smaller group, also missing an edge
				// the other vertex and reachableIndex can not share an edge yet,
				// otherwise they would be in the same group
				for (int reachableIndex : reachableVertices) {
					final FeatureVector reachableFeature = graph.getVertexById(reachableIndex).getFeature();					
					for (int otherIndex : otherVertices) {
						final FeatureVector otherFeature = graph.getVertexById(otherIndex).getFeature();
						final float distance = space.computeDistance(otherFeature, reachableFeature); 

						if(distance < bestDistance) {
							bestDistance = distance;
							bestOtherVertex = otherIndex;
							bestReachableVertex = reachableIndex;
						}
					}
				}

				// connect reachableIndex and otherIndex
				graph.addUndirectedEdge(bestReachableVertex, bestOtherVertex, bestDistance);
				changes.add(new BuilderChange(bestReachableVertex, bestOtherVertex, bestDistance, false));

				// move the element from the list of missing edges
				reachableGroup.hasEdge(bestReachableVertex);
				otherGroup.hasEdge(bestOtherVertex);

				// merge both groups
				otherGroup.merge(reachableGroup);

				// remove the current group from the list of group since its merged
				reachableGroups.remove(reachableGroups.size()-1);	        	
			}

			// 3.2 now all groups are reachable but still some vertices are missing edge, try to connect them to each other.
			final VertexData[] remainingVertices = reachableGroups.get(0).getMissingEdges().stream().map(graph::getVertexById).toArray(VertexData[]::new);
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

						if(label == 591798 && (vertexA.getId() == 157255 || bestVertexBId == 157255))
							System.out.println("overflow");
					}
				}
			}


			// 3.3 the remaining vertices can not be connected to any of the other involved vertices, because they already have an edge to all of them.
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

							if(label == 591798 && (vertexA.getId() == 157255 || vertexB.getId() == 157255))
								System.out.println("overflow");

							if(label == 591798 && (vertexC.getId() == 157255 || vertexD.getId() == 157255))
								System.out.println("overflow");

							break;
						}
					}
				}
			}
		}

		for (int involvedVertex : involvedVertices) {
			if(graph.getVertexById(involvedVertex).getEdges().size() != edgesPerVertex)
				System.out.println(involvedVertex+" has too many edges "+graph.getVertexById(involvedVertex).getEdges().size());
		}

	}


	protected static class UnionFind  {

		protected final IntIntMap parents;

		public UnionFind(int expectedSize) {
			parents = HashIntIntMaps.newMutableMap(expectedSize);
		}

		/**
		 * Find the root of the set in which element belongs
		 * 
		 * @param element
		 * @return
		 */
		public int find(int element) {
			final int entry = parents.getOrDefault(element, -1);
			if (entry == element) // if element is root
				return element;
			return find(entry); // recurs for parent till we find root
		}

		/**
		 * If the parents are known via find this method can be called instead of union
		 * 
		 * @param parent1
		 * @param parent2
		 */
		public void update(int parent1, int parent2) {
			parents.put(parent1, parent2);
		}
	}

	protected static class ReachableGroup implements Iterable<Integer> {

		protected final int vertexIndex;
		protected final Set<Integer> missingEdges = new HashSet<>();
		protected final Set<Integer> reachableVertices = new HashSet<>();

		public ReachableGroup(int vertexIndex) {
			this.vertexIndex = vertexIndex;
			this.missingEdges.add(vertexIndex);
			this.reachableVertices.add(vertexIndex);
		}

		public int getVertexIndex() {
			return vertexIndex;
		}

		public Set<Integer> getMissingEdges() {
			return missingEdges;
		}

		public int size() { 
			return reachableVertices.size();
		}

		@Override
		public int hashCode() {
			return getVertexIndex();
		}

		@Override
		public String toString() {
			return "Group of "+vertexIndex+" with "+reachableVertices.size() +" vertices " + reachableVertices.stream().map(n -> String.valueOf(n)).collect(Collectors.joining(", ", "{", "}"));
		}

		/**
		 * Merge both groups vertex sets and let both reference the same set
		 * 
		 * @param otherGroup
		 */
		public void merge(ReachableGroup otherGroup) {

			// skip if both are the same object
			if(getVertexIndex() == otherGroup.getVertexIndex())
				return;

			// merge
			missingEdges.addAll(otherGroup.missingEdges);
			reachableVertices.addAll(otherGroup.reachableVertices);
		}

		public void hasEdge(int vertexIndex) {
			missingEdges.remove(vertexIndex);
		}

		@Override
		public Iterator<Integer> iterator() {
			return reachableVertices.iterator();
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
			if(this.newEntryQueue.isEmpty() == false || this.removeEntryQueue.isEmpty() == false) {
				long addTaskManipulationIndex = Long.MAX_VALUE;
				long delTaskManipulationIndex = Long.MAX_VALUE;

				if(this.newEntryQueue.isEmpty() == false) 
					addTaskManipulationIndex = this.newEntryQueue.peek().manipulationIndex;

				if(this.removeEntryQueue.isEmpty() == false) 
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
