package com.vc.deg.ref.hierarchy;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.graph.GraphDesigner;
import com.vc.deg.graph.VertexCursor;
import com.vc.deg.graph.VertexFilter;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner.BuilderAddTask;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner.BuilderRemoveTask;

public class HierarchicalGraphDesigner implements GraphDesigner {
	
	protected static final float desiredShrinkFactor = 4;
	
	
	protected final List<DynamicExplorationGraph> layers;
	protected final Map<Integer, Integer> labelToRank;
	protected final FeatureSpace space;
	protected final int edgesPerVertex;
	protected final int topRankSize;
	
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
	
	public HierarchicalGraphDesigner(List<DynamicExplorationGraph> layers, FeatureSpace space, int edgesPerVertex, int topRankSize) {
		this.layers = layers;
		this.labelToRank = new HashMap<>();
		this.space = space;
		this.edgesPerVertex = edgesPerVertex;
		this.topRankSize = topRankSize;
		
		this.manipulationCounter = new AtomicLong(0);
		this.newEntryQueue = new ConcurrentLinkedQueue<>();
		this.removeEntryQueue = new ConcurrentLinkedQueue<>();
		
		ensureRank(1);
		
		// default hyper parameters
		setRandom(new Random(7));
		setExtendK(edgesPerVertex * 2);
		setExtendEps(0.2f);
		setExtendSchema(true);
		setImproveK(edgesPerVertex);
		setImproveEps(0.001f);
		setMaxPathLength(5);

		// fill the rank map
		for (int i = 0; i < layers.size(); i++) {
			final int rank = i;
			final VertexCursor cursor = layers.get(i).vertexCursor();
			while(cursor.moveNext())
				this.labelToRank.put(cursor.getVertexLabel(), rank);
		}		
	}
	
	
	/**
	 * 
	 * @param graph
	 * @return
	 */
	protected DynamicExplorationGraph applyDesignerSettings(DynamicExplorationGraph graph) {
		graph.designer().setRandom(rnd);
		graph.designer().setExtendK(extendK);
		graph.designer().setExtendEps(extendEps);
		graph.designer().setImproveK(improveK);
		graph.designer().setImproveEps(improveEps);
		graph.designer().setMaxPathLength(maxPathLength);
		return graph;
	}
	
	// ----------------------------------------------------------------------------------------------
	// ----------------------------------- hyper parameters -----------------------------------------
	// ----------------------------------------------------------------------------------------------

	@Override
	public void setRandom(Random rnd) {
		this.rnd = rnd;
		for (DynamicExplorationGraph graph : layers) 
			graph.designer().setRandom(rnd);
	}

	@Override
	public void setExtendK(int k) {
		this.extendK = k;
		for (DynamicExplorationGraph graph : layers) 
			graph.designer().setExtendK(extendK);
	}

	@Override
	public void setExtendEps(float eps) {
		this.extendEps = eps;
		for (DynamicExplorationGraph graph : layers) 
			graph.designer().setExtendEps(extendEps);
	}

	@Override
	public void setExtendSchema(boolean useSchemaC) {
		schemaC = useSchemaC;
		for (DynamicExplorationGraph graph : layers) 
			graph.designer().setExtendSchema(schemaC);
	}

	@Override
	public void setImproveK(int k) {
		this.improveK = k;
		for (DynamicExplorationGraph graph : layers) 
			graph.designer().setImproveK(improveK);
	}

	@Override
	public void setImproveEps(float eps) {
		this.improveEps = eps;
		for (DynamicExplorationGraph graph : layers) 
			graph.designer().setImproveEps(improveEps);
	}

	@Override
	public void setMaxPathLength(int maxPathLength) {
		this.maxPathLength = maxPathLength;
		for (DynamicExplorationGraph graph : layers) 
			graph.designer().setMaxPathLength(maxPathLength);
	}



	// ----------------------------------------------------------------------------------------------
	// ----------------------------------- evaluation methods ---------------------------------------
	// ----------------------------------------------------------------------------------------------

	@Override
	public float calcAvgEdgeWeight() {
		float avg = 0;
		for (DynamicExplorationGraph graph : layers) 
			avg += graph.designer().calcAvgEdgeWeight();
		return avg/layers.size();
	}
	
	@Override
	public float calcAvgNeighborRank() {
		float avg = 0;
		for (DynamicExplorationGraph graph : layers) 
			avg += graph.designer().calcAvgNeighborRank();
		return avg/layers.size();
	}

	@Override
	public float calcAvgNeighborRank(int[][] topList) {
		float avg = 0;
		for (DynamicExplorationGraph graph : layers) 
			avg += graph.designer().calcAvgNeighborRank(topList);
		return avg/layers.size();
	}
	

	@Override
	public boolean checkGraphValidation(int expectedVertices, int expectedNeighbors) {
		boolean valid = layers.get(0).designer().checkGraphValidation(expectedVertices, expectedNeighbors);
		
		int vertexCount = topRankSize;
		for (int i = 1; i < layers.size() && valid; i++) {
			valid = layers.get(layers.size() - i).designer().checkGraphValidation(vertexCount, expectedNeighbors);
			vertexCount *= desiredShrinkFactor;
		}
		return valid;
	}
	
	
	// ----------------------------------------------------------------------------------------------
	// ----------------------------------- hierarchy methods ----------------------------------------
	// ----------------------------------------------------------------------------------------------
	
	/**
	 * Compute the rank distribution for a specific amount of elements.
	 * The max grow factor decides how much more elements are on a rank
	 * above. While the top rank has a maximum size as well.
	 * 
	 * All ranks from top to bottom have a grow factor equal to the max
	 * grow factor. Only rank 0 has a smaller factor, since it is always
	 * the given size.
	 * 
	 * @param size
	 * @param topRankSize
	 * @param maxGrowFactor
	 * @return
	 */
	protected static int[] rankDistribution(int size, int topRankSize, float maxGrowFactor) {
		
		// how many ranks are needed
		int desiredRanks = (int) Math.max(1, 1 + Math.ceil(Math.log(size / (float)topRankSize) / Math.log(maxGrowFactor)));

		// element count at each rank (with a fix top rank and all elements at rank 0)
		int[] rankSize = new int[desiredRanks];
		rankSize[0] = size;
		if(desiredRanks > 1) {
			rankSize[desiredRanks-1] = topRankSize;
			for (int i = desiredRanks-2; i > 0; i--) {
				rankSize[i] = (int)(rankSize[i+1] * maxGrowFactor);
			}
		}
		
		return rankSize;
	}
	
	/**
	 * Ensures there is enough space and ranks. 
	 * Adds a rank if needed and promotes existing elements.
	 * 
	 * @param size
	 */
	protected void ensureSize(int size) {
		
		// how many elements are at each rank
		int[] rankSizes = rankDistribution(size, topRankSize, desiredShrinkFactor);
		
		// not enough ranks yet	
		ensureRank(rankSizes.length);
	}
	
	/**
	 * Ensures there is enough ranks.
	 * 
	 * @param rankCount
	 */
	protected void ensureRank(int rankCount) {
		
		// not enough ranks yet
		while(layers.size() < rankCount) {
			
			// add another rank at the bottom (new rank 0) and fill it with the keys from the rank above
			final DynamicExplorationGraph deg = (layers.size() == 0) ? new com.vc.deg.ref.DynamicExplorationGraph(space, edgesPerVertex) : layers.get(0).copy();
			applyDesignerSettings(deg);
			layers.add(0, deg);
						
			// promote all existing elements one rank up
			for(Map.Entry<Integer, Integer> entry : labelToRank.entrySet()) 
				entry.setValue(entry.getValue() + 1);
		}
	}
	

	/**
	 * Balances the ranks if a element has been removed
	 */
	protected void balanceRanks() {

		// how many elements are at each rank
		final int[] rankSizes = rankDistribution(labelToRank.size(), topRankSize, desiredShrinkFactor);

		// promote entire rank 0 to rank 1 and remove rank 0 afterwards
		while(rankSizes.length < layers.size()) {
			
			// just remove rank 1 and it is done. 
			// Reason: Promoting all elements at rank 0 to rank 1 creates just a copy of rank 0 at rank 1.
			layers.remove(1);
			
			// degrade all existing elements one rank down
			for(Map.Entry<Integer, Integer> entry : labelToRank.entrySet()) 
				entry.setValue(Math.max(0, entry.getValue() - 1));
		}
		
		// promote ids to higher graph ranks if needed
		for (int rank = 1; rank < rankSizes.length; rank++) {			
			final DynamicExplorationGraph higherLayer = layers.get(rank);	
			final DynamicExplorationGraph lowerLayer = layers.get(rank-1);
			
			// find vertices to promote until the current rank is big enough
			final int requiredAdds = rankSizes[rank] - higherLayer.size();
			final Map<Integer, FeatureVector> newVertices = new HashMap<>();
			while(newVertices.size() < requiredAdds) {
				
				// get random node from a rank below which does not exist at the current rank
				int label = getRandomLabel(lowerLayer, rank-1);
				
				newVertices.put(label, lowerLayer.getFeature(label));
			}
			
			// add the vertices to promote to the next layer
			if(newVertices.size() > 0) {
				final GraphDesigner designer = higherLayer.designer();
				
				// add the vertex to the designer
				for (Map.Entry<Integer, FeatureVector> entry : newVertices.entrySet()) {					
					designer.add(entry.getKey(), entry.getValue());
					labelToRank.put(entry.getKey(), rank);
				}

				// start the design process to add the vertices to the graph
				designer.build((long step, long added, long removed, long improved, long tries, int lastAdd, int lastRemoved) -> {
					if(lastAdd == -1 && lastAdd == -1)
						designer.stop();
				});
			}
		}		
	}
		
	/**
	 * Get a random id which only exists on this rank and below.
	 * 
	 * @param targetRank
	 * @return
	 */
	protected int getRandomLabel(int targetRank) {
		return getRandomLabel(layers.get(targetRank), targetRank);
	}
	
	/**
	 * Get a random id which only exists on this rank and below.
	 * 	
	 * @param graph
	 * @param targetRank
	 * @return -1 = no valid id can be found
	 */
	protected int getRandomLabel(DynamicExplorationGraph graph, int targetRank) {	
		// TODO build a data structure which contains a set of ids for each rank of the graph. the set only has ids which exists only on this rank or below.
		// TODO use layers.get(targetRank).labelFilter() instead of the new filter object
		return graph.getRandomLabel(rnd, new VertexFilter() {
			
			@Override
			public int size() {
				return graph.size() - (int) ((float)graph.size() / desiredShrinkFactor);
			}
			
			@Override
			public float getInclusionRate() {
				return Math.max(0, Math.min(1, ((float)size()) / graph.size()));
			}
			
			@Override
			public boolean isValid(int label) {
				return labelToRank.get(label) == targetRank;
			}
			
			@Override
			public void forEachValidId(IntConsumer action) {
			}
		});
	}
	
	// ----------------------------------------------------------------------------------------------
	// ----------------------------------- manipulation methods -------------------------------------
	// ----------------------------------------------------------------------------------------------

	@Override
	public void add(int label, FeatureVector data) {
		
		// check the feature vector compatibility
		final FeatureSpace featureSpace = layers.get(0).getFeatureSpace();
		if(data.dims() != featureSpace.dims() || data.getComponentType() != featureSpace.getComponentType())
			throw new RuntimeException("Invalid data component type "+data.getComponentType().getSimpleName()+" or dimension "+data.dims()+
									   ", expected "+featureSpace.getComponentType().getSimpleName()+" and "+featureSpace.dims());
		
		newEntryQueue.offer(new BuilderAddTask(label, manipulationCounter.getAndIncrement(), data));
	}

	@Override
	public void remove(int label) {
		removeEntryQueue.offer(new BuilderRemoveTask(label, manipulationCounter.getAndIncrement()));
	}
	
	@Override
	public void removeIf(IntPredicate filter) {
		labelToRank.keySet().forEach(id -> {
			if(filter.test(id))
				remove(id);
		});
	}
	
	/**
	 * Add a vertex to the hierarchical graph. Add layer if needed.
	 * 
	 * @param label
	 * @param feature
	 */
	private boolean extendGraph(int label, FeatureVector feature) {
		if(labelToRank.containsKey(label) == false) {
			int newSize = labelToRank.size() + 1;
			
			// make sure there is enough space to add another element
			ensureSize(newSize);
			
			// compute chance to upgrade the new element at a higher rank
			final int maxRank = layers.size() - 1;
			int targetRank = 0;
			for (int r = maxRank; r > 0; r--) {
				if(rnd.nextFloat() < ((float)layers.get(r).size() / newSize)) {
					targetRank = r;
					break;
				}
			}
			
			// add the new element at the rank and all ranks below
			// degrades other existing elements from the ranks to make space
			for (int r = targetRank; r > 0; r--) {
				
				// degrade random key which only exists on this rank
				int keyToDegrade = getRandomLabel(r);
				removeVertexFromGraphAtRank(keyToDegrade, r);
				labelToRank.put(keyToDegrade, r - 1);

				// add new key at rank
				addVertexToGraphAtRank(label, feature, r);
			}
			
			// add new key to rank 0 and map of all keys
			addVertexToGraphAtRank(label, feature, 0);
			labelToRank.put(label, targetRank);
			
			return true;
		}
		throw new RuntimeException("Label:"+label+" already exists in the graph");
		//return false;
	}

	/**
	 * Add a vertex to a graph at a specific rank
	 * 
	 * @param label
	 * @param feature
	 * @param rank
	 */
	private void addVertexToGraphAtRank(int label, FeatureVector feature, int rank) {
		final GraphDesigner graphDesigner = layers.get(rank).designer();
		graphDesigner.add(label, feature);
		graphDesigner.build((long step, long added, long removed, long improved, long tries, int lastAdd, int lastRemoved) -> {
			if(lastAdd == -1 && lastAdd == -1)
				graphDesigner.stop();
		});
	}
	

	
	/**
	 * Remove a vertex from the entire graph and all its hierarchy layers.
	 * Delete layers if needed.
	 * 
	 * @param label
	 */
	private boolean shrinkGraph(int label) {
		if(labelToRank.containsKey(label) == true) {
			
			// remove key from all ranks
			int maxRank = labelToRank.remove(label);
			for (int rank = 0; rank <= maxRank; rank++) 
				removeVertexFromGraphAtRank(label, rank);
			
			// balance the ranks
			balanceRanks();
			
			return true;
		}
		throw new RuntimeException("Label:"+label+" does not exists in the graph");
	}
	
	/**
	 * Remove a vertex from a graph at a specific rank
	 * 
	 * @param label
	 * @param rank
	 */
	private void removeVertexFromGraphAtRank(int label, int rank) {
		final GraphDesigner graphDesigner = layers.get(rank).designer();
		graphDesigner.remove(label);
		graphDesigner.build((long step, long added, long removed, long improved, long tries, int lastAdd, int lastRemoved) -> {
			if(lastAdd == -1 && lastAdd == -1)
				graphDesigner.stop();
		});
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

		try {		
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
						final BuilderAddTask addTask = this.newEntryQueue.poll();
						if(extendGraph(addTask.label, addTask.feature)) {
							added++;
							lastAdd = addTask.label;
						}
					} else {
						final BuilderRemoveTask removeTask = this.removeEntryQueue.poll();
						if(shrinkGraph(removeTask.label)) {
							deleted++;
							lastDelete = removeTask.label;
						}
					}
				}
	
				step++;
	
				// inform the listener
				listener.onChange(step, added, deleted, improved, tries, lastAdd, lastDelete);
				lastAdd = -1; 
				lastDelete = -1;
			}
		} catch (Exception e) {
			stopBuilding = true;
			throw new RuntimeException("An error occured during graph construction, process stopped.", e);
		}
	}

	@Override
	public void stop() {
		stopBuilding = true;
	}
}