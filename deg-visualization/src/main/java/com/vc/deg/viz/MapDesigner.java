package com.vc.deg.viz;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;
import java.util.function.IntFunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.koloboke.collect.set.IntSet;
import com.koloboke.collect.set.hash.HashIntSets;
import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.HierarchicalDynamicExplorationGraph;
import com.vc.deg.graph.GraphFilter;
import com.vc.deg.graph.VertexCursor;
import com.vc.deg.viz.feature.FeatureTransformer;
import com.vc.deg.viz.filter.MutableGraphFilter;
import com.vc.deg.viz.filter.PreparedGraphFilter;
import com.vc.deg.viz.model.GridMap;
import com.vc.deg.viz.model.MotionVector;
import com.vc.deg.viz.model.WorldMap;
import com.vc.deg.viz.om.FastLinearAssignmentSorterAdapter;


/**
 * Projects parts of a graph on a 2D grid.
 * 
 * @author Nico Hezel
 */
public class MapDesigner {

	protected static final Logger log = LoggerFactory.getLogger(MapDesigner.class);
	
	public static final int MEDIAN_ELEMENT_COUNT = 3;
	
	protected final HierarchicalDynamicExplorationGraph graph;	
	protected final FastLinearAssignmentSorterAdapter flas;

	public MapDesigner(HierarchicalDynamicExplorationGraph graph) {
		this(graph, FeatureTransformer.findTransformer(graph.getFeatureSpace()), id -> graph.getFeature(id));
	}
		
	public MapDesigner(HierarchicalDynamicExplorationGraph graph, FeatureTransformer transformer, IntFunction<FeatureVector> idToFeature) {
		this(graph, id -> transformer.transform(idToFeature.apply(id)));
	}
	
	public MapDesigner(HierarchicalDynamicExplorationGraph graph, IntFunction<float[]> idToFloatFeature) {		
		this.flas = new FastLinearAssignmentSorterAdapter(idToFloatFeature);
		this.graph = graph;
	}
	
	/**
	 * Create a new graph filter with all the ids from the graph level minus the id from the world map
	 * or use the existing filter and remove the ids from the world map
	 * 
	 * @param globalFilter can be null
	 * @param worldMap can be null
	 * @param atLevel
	 * @return
	 */
	protected MutableGraphFilter filterAtLevel(final MutableGraphFilter globalFilter, WorldMap worldMap, int atLevel) {
		
		final DynamicExplorationGraph deg = graph.getGraph(atLevel);
		final IntSet validIds;
		
		if(globalFilter == null) {
			validIds = HashIntSets.newMutableSet(c -> {
				final VertexCursor cursor = deg.vertexCursor();
				while(cursor.moveNext())
					c.accept(cursor.getVertexLabel());			
			}, deg.size());
		
		} else {
			
			// the globalFilter contains valid ids of graph level 0
			if(atLevel == 0) {
				
				// make a copy of the global filter and remove all ids which are on the world map from the copy
				if(worldMap != null)
					return globalFilter.remove(worldMap::foreachCell);
				
				return globalFilter;
			}
			
			// performance optimization: Intersection of valid filter and graph ids. 
			// 							 Iterate over the smaller one to reduce the number of cache misses.
			if(globalFilter.size() < deg.size()) {
				validIds = HashIntSets.newMutableSet(c -> {
					globalFilter.forEachValidId((int label) -> {
						if(deg.hasLabel(label))
							c.accept(label);
					});
				}, globalFilter.size());
			} else {				
				validIds = HashIntSets.newMutableSet(c -> {
					final VertexCursor cursor = deg.vertexCursor();
					while(cursor.moveNext()) {
						final int label = cursor.getVertexLabel();
						if(globalFilter.isValid(label))
							c.accept(label);
					}
				}, deg.size());
			}			
		}

		// remove all ids which are on the world map from the list of valid ids
		if(worldMap != null) {
			worldMap.foreachCell(cell -> {
				validIds.removeInt(cell);
			});
		}

		return new PreparedGraphFilter(validIds);
	}
	
	/**
	 * Fill the map with the elements from the world map. If there are still holes fill them with the neighboring elements.
	 * In the case there are no elements at all an empty map will be returned.
	 * 
	 * @param localMap
	 * @param worldPosX (world coordinates)
	 * @param worldPosY (world coordinates)
	 * @param atLevel
	 * @param globalFilter can be null
	 */
	public void fill(WorldMap worldMap, GridMap localMap, int worldPosX, int worldPosY, int atLevel, MutableGraphFilter globalFilter) {
		
		// no graph is present yet
		if(graph == null) 
			return;
		
		// copy the cell data from the world map to the local map
		worldMap.copyTo(localMap, worldPosX, worldPosY);
		
		// find neighbors in the graph for the free cells of the map 
		int emptyTileCount = localMap.freeCount();
		if(emptyTileCount > 0 && localMap.isEmpty() == false) {
			
			// list of used and unused cells[y][x]
			final boolean[][] inUse = getUsedPositions(localMap);
			
			// get all border images ignore there order
			final IntFloat[] borderPanelPos = getBorderPositions(localMap, new int[] {0,0});
			final int[] queryImages = Arrays.stream(borderPanelPos).mapToInt(p -> localMap.get(p.getIndex())).toArray();
			
			// get the most similar neighbors of the border images from the graph
			long start = System.currentTimeMillis();
			final int[] images = getBestNeighborVertices(queryImages, atLevel, emptyTileCount, filterAtLevel(globalFilter, worldMap, atLevel));
			
			// fill the new neighbors into the free cells of the map
			final int fillCount = fillFreePlaces(localMap, images);
			
			// if new images where placed onto the map
			if(fillCount > 0) {
				log.debug("Fill free tiles ("+images.length+" elements) took "+(System.currentTimeMillis() - start)+"ms");
				
				// arrange them
				start = System.currentTimeMillis();
				arrangeMap(localMap, inUse);
				log.debug("Arranging the neighbourhood took "+(System.currentTimeMillis() - start)+"ms");

				// copy the new local map to the world map
				worldMap.copyFrom(localMap, worldPosX, worldPosY);
			}
		}
	}
	
	/**
	 * Jump in the graph to a position most similar to the targetImage. Place the image at the target position in the map.
	 * Arrange other similar images from the graph around the image on the map. The targetImage might be not visible if it
	 * does not pass the filter. Use the graph at given the level. 
	 * 
	 * @param worldMap
	 * @param localMap
	 * @param targetElementId
	 * @param targetPosX (local coordinates)
	 * @param targetPosY (local coordinates)
	 * @param worldPosX (world coordinates)
	 * @param worldPosY (world coordinates)
	 * @param atLevel
	 * @param globalFilter can be null
	 */
	public void jump(WorldMap worldMap, GridMap localMap, int targetElementId, int targetPosX, int targetPosY, int worldPosX, int worldPosY, int atLevel, MutableGraphFilter globalFilter) {
		
		// do nothing if the graph is not available
		if(graph == null) 
			return;

		// prepare filter of valid element ids
		final long start = System.currentTimeMillis();
		final MutableGraphFilter filterAtLevel = filterAtLevel(globalFilter, null, atLevel);
				
		// if the number of valid elements is too small just arrange them all at once on the world map
		if(filterAtLevel.size() < 1000) {
			
			// does the target element pass the filter or should it be ignored
			final int targetElement = (globalFilter != null && globalFilter.isValid(targetElementId) == false) ? -1 : targetElementId;
						
			// copy the element ids from the filter to an array, ignore the id identical to the target element
			final int[] neighbors = new int[filterAtLevel.size() - (filterAtLevel.isValid(targetElement) ? 1 : 0)];
			final int[] pos = new int[1];
			filterAtLevel.forEachValidId(id -> {
				if(id != targetElement)
					neighbors[pos[0]++] = id;
			});
			
			// arrange on a bigger map first and then copy parts of it to the local map
			final GridMap map = findMapSize(neighbors.length + ((targetElement >= 0) ? 1 : 0));
			final int mapCenterX = map.columns() / 2;
			final int mapCenterY = map.rows() / 2;
			arrange(worldMap, map, targetElement, mapCenterX, mapCenterY, targetPosX-mapCenterX, targetPosY-mapCenterY, neighbors);
			worldMap.copyTo(localMap, worldPosX, worldPosY);
			log.debug("Collecting and arranging "+neighbors.length+" elements took "+(System.currentTimeMillis() - start)+"ms");

		} else {
			
			// gather the most similar vertices from the neighborhood of the target element
			final int[] neighbors = getBestNeighborVertices(new int[] {targetElementId}, atLevel, localMap.size(), filterAtLevel);
			
			// does the target image pass the filter or should it be ignored
			final int targetElement = (globalFilter != null && globalFilter.isValid(targetElementId) == false) ? -1 : targetElementId;
			
			// arrange the images onto the local map
			arrange(worldMap, localMap, targetElement, targetPosX, targetPosY, worldPosX, worldPosY, neighbors);
			log.debug("Collecting and arranging "+neighbors.length+" elements took "+(System.currentTimeMillis() - start)+"ms");
		}
	}
	
	
	/**
	 * Find a width/height combination which does have as few holes as possible and is close to quadratic.
	 * 
	 * @return the grid map of the width and height
	 */
	protected static GridMap findMapSize(int numOfElements) {
		int width = (int) Math.sqrt(numOfElements+1);
		int height = width;
		
		while (width * height < numOfElements) 
			width++;
		
		return new GridMap(width, height);
	}
	
	/**
	 * Arranges the elements on the local map.
	 * Stores the arrangement in the world map at the given position.
	 * 
	 * The provided elements do not have to be in the graph in order 
	 * to be arranged.
	 * 
	 * Place the targetImage at the target position in the map if 
	 * its is a valid id (id higher or equal zero)
	 * 
	 * @param worldMap
	 * @param localMap
	 * @param targetImageId
	 * @param targetPosX
	 * @param targetPosY
	 * @param worldPosX
	 * @param worldPosY
	 * @param elements
	 */
	protected void arrange(WorldMap worldMap, GridMap localMap, int targetImageId, int targetPosX, int targetPosY, int worldPosX, int worldPosY, int[] elements) {
		final long start = System.currentTimeMillis();
		
		// place the most similar images onto the local map from best to worst
		localMap.clear();
		for (int i = 0; i < elements.length; i++) 
			localMap.set(i, elements[i]);
		
		// move the image at the target position to the last place on the map 
		// override the last map place  in the process (least similar image)
		// place the target image id at the target position if its a valid id
		if(targetImageId >= 0) {
			final int targetTile = localMap.get(targetPosX, targetPosY);
			localMap.set(localMap.columns() - 1 , localMap.rows() - 1, targetTile);
			localMap.set(targetPosX, targetPosY, targetImageId);
		}
				
		// block the target position from being swapped in the following arrangement process
		final boolean[][] blockedPanels = new boolean[localMap.rows()][localMap.columns()];
		if(targetImageId >= 0) 
			blockedPanels[targetPosY][targetPosX] = true;
		
		// arrange all non blocked map cells by their image similarity of its spatial neighbors
		arrangeMap(localMap, blockedPanels);
		
		// copy the arrangement onto the world map
		worldMap.copyFrom(localMap, worldPosX, worldPosY);
		log.debug("Arranging the neighbourhood took "+(System.currentTimeMillis() - start)+"ms");
	}

	
	/**
	 * Move the map a few steps, defined by the shift vector and fill along the direction vector direction.
	 * Shift and direction vector can point in slightly different directions, 
	 *  
	 * @param worldMap
	 * @param localMap
	 * @param shiftVector (world coordinates)
	 * @param directionVector (direction vector)
	 * @param worldPosX
	 * @param worldPosY
	 * @param atLevel
	 * @param globalFilter can be null
	 */
	public void move(WorldMap worldMap, GridMap localMap, MotionVector shiftVector, MotionVector directionVector, int worldPosX, int worldPosY, int atLevel, MutableGraphFilter globalFilter) {
		
		// gibt es überhaupt eine Verschiebung oder einen graph
		if(graph == null || shiftVector.length() == 0) 
			return;
		
		// kopiere existierende Panels von der Weltkarte auf die lokale Karte
		worldMap.copyTo(localMap, worldPosX, worldPosY);

		// sind alle Plätze auf diesem Teil der Karte bereits belegt 
		int emptyTileCount = localMap.freeCount();
		if(emptyTileCount > 0 && localMap.isEmpty() == false) {
			
			// welche Plätze sind schon belegt
			boolean[][] inUse = getUsedPositions(localMap);
			
			// fülle die leeren Felder falls notwendig
			long start = System.currentTimeMillis();
			int[] images = getNeighbors(localMap, shiftVector, directionVector, atLevel, filterAtLevel(globalFilter, worldMap, atLevel));
			
			// fülle die fehlenden Felder mit den Nachbarn der eben besorgten Bilder
			int fillCount = fillFreePlaces(localMap, images);

			// es sind neue element hinzu gekommen
			if(fillCount > 0) {
				log.debug("Fill by shift vector ("+fillCount+" elements) took "+(System.currentTimeMillis() - start)+"ms");
				
				// sortiere alle Bilder
				start = System.currentTimeMillis();
				arrangeMap(localMap, inUse);
				log.debug("Arranging the neighbourhood took "+(System.currentTimeMillis() - start)+"ms");

				// die Sortierung auf die Weltkarte kopieren
				worldMap.copyFrom(localMap, worldPosX, worldPosY);
			}
		}
	}
	
	/**
	 * Acquire the neighbor images from the graph of all the image where the shift vector points to on the map.
	 * 
	 * @param localMap
	 * @param shiftVector
	 * @param directionVector
	 * @param atLevel
	 * @param globalFilter
	 * @return number of new elements
	 */
	protected int[] getNeighbors(GridMap localMap, MotionVector shiftVector, MotionVector directionVector, int atLevel, GraphFilter globalFilter) {		
		final float rows = localMap.rows();
		final float columns = localMap.columns();
		
		// extend the direction vector until it hits the a tile at the border
		final MotionVector nDirection = directionVector.normalize();
		final float cdx = (int)(nDirection.getX() * (columns/2f - Math.abs(shiftVector.getX())));
		final float cdy = (int)(nDirection.getY() * (rows/2f - Math.abs(shiftVector.getY())));
		
		// position of the tile hit by the direction vector
		int arrayPosX = Math.round(columns/2f - cdx); 
		int arrayPosY = Math.round(rows/2f    - cdy);
		if(shiftVector.getY() > 0) arrayPosY--;
		if(shiftVector.getX() > 0) arrayPosX--;
		final int[] panelCoordinates = new int[] {arrayPosX, arrayPosY};
						 
		// get the images along the local map border near the hit tile
		IntFloat[] borderPanelPos = getBorderPositions(localMap, panelCoordinates);
		borderPanelPos = Arrays.copyOf(borderPanelPos, Math.max(1, borderPanelPos.length/2)); // ignore the ones far away
		final int[] queryImages = Arrays.stream(borderPanelPos).mapToInt(p -> localMap.get(p.getIndex())).toArray();
		 
		// find neighbors in the graph not present on the world map and pass the filter
		final int desiredCount = localMap.freeCount();	
		return getBestNeighborVerticesMedian(queryImages, atLevel, desiredCount, globalFilter);
	}

	/**
	 * Sort the elements on the local map
	 * 
	 * @param localMap
	 * @param blockedPlaces[y][x]
	 */
	protected void arrangeMap(GridMap localMap, boolean[][] blockedPlaces) {		
		try {		
			flas.arrangeWithHoles(localMap, blockedPlaces);			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	
	/**
	 * Find the three median images from the list of query images and find their best neighbors with {@link #getBestNeighborVertices(int[], int, int, GraphFilter)}}
	 * 
	 * @param queryImages
	 * @param atLevel
	 * @param desiredCount
	 * @param globalFilter
	 * @return
	 */
	public int[] getBestNeighborVerticesMedian(int[] queryImages, int atLevel, int desiredCount, GraphFilter globalFilter) {
		
		if(desiredCount == 0) 
			return new int[0];
		
		
		final FeatureSpace visualFeatureSpace = graph.getFeatureSpace();
		final FeatureVector[] queryFeatures = Arrays.stream(queryImages).mapToObj(q -> graph.getFeature(q)).toArray(FeatureVector[]::new);
		
		// jedes Query Bild gegen jedes andere Query Bild
		final List<IntFloat> sumOfDistances = new ArrayList<>();
		for (int i = 0; i < queryImages.length; i++) {
			final FeatureVector queryFeature = queryFeatures[i];
			
			// Summe der Distanzen zu allen query Bildern
			final float distanceSum = (float)Arrays.stream(queryFeatures).mapToDouble(fv -> {
				return visualFeatureSpace.computeDistance(fv, queryFeature);
			}).sum();
			
			sumOfDistances.add(new IntFloat(i, distanceSum));			
		}
		
		// list von median Bildern
		final int[] medianImages = sumOfDistances.stream()
												 .sorted(IntFloat.asc())
												 .limit(MEDIAN_ELEMENT_COUNT)
												 .mapToInt(d -> queryImages[d.getIndex()])
												 .toArray();		
		
		// normale Suche mit den Median Bildern
		return getBestNeighborVertices(medianImages, atLevel, desiredCount, globalFilter);
	}
	
	/**
	 * Find the most similar neighbors of the query images. Only images passing the 
	 * filter are allowed and the max similarity decides which to chose.
	 * 
	 * If a query image is not available at the given graph level, the most similar 
	 * will be used.
	 * 
	 * The query images are not included in the result list. The order is ascending 
	 * from the most similar to the least similar.
	 * 
	 * @param queryImageIds
	 * @param atLevel
	 * @param desiredCount
	 * @param filter
	 * @return
	 */
	protected int[] getBestNeighborVertices(int[] queryImageIds, int atLevel, int desiredCount, GraphFilter filter) {
		
		// not enough images or invalid goal
		if(desiredCount <= 0 || queryImageIds.length == 0) 
			return new int[0];
				
		// search for similar query images at the given level, for all query images which are not present at the level
		final int[] queryIds = HashIntSets.newImmutableSet(c -> {			
			for (int queryImageId : queryImageIds) {
				c.accept(getSimilarVertexAtLevel(queryImageId, atLevel, 1, 0.6f));
			}
		}, queryImageIds.length).toIntArray();
		
		// explore from the query ids other vertices in the graph which are valid in the filter
		final int[] exploreResult = graph.exploreAtLevel(queryIds, atLevel, desiredCount, Integer.MAX_VALUE, filter);


		
		
		// edge case: One of the initial query image ids where not present on the current level and an alternative
		// was stored in the queryIds array. This alternative is not in exploreResult but might pass the filter and
		// be better than any image in the exploreResult list.
		final TreeSet<IntFloat> result = new TreeSet<>(IntFloat.asc());
		{
			// get the feature of the query ids
			final FeatureSpace space = graph.getFeatureSpace();
			final FeatureVector[] queryFeature = new FeatureVector[queryIds.length];
			for (int i = 0; i < queryFeature.length; i++) 
				queryFeature[i] = graph.getFeature(queryIds[i]);
			
			// a function to compute the smallest distance to any of the queries
			final IntFunction<IntFloat> calcMinDistance = (int id) -> {
				final FeatureVector fv = graph.getFeature(id);
				
				float minDistance = Float.MAX_VALUE;
				for (FeatureVector query : queryFeature) {
					final float dist = space.computeDistance(query, fv);
					if(dist < minDistance) 
						minDistance = dist;
				}
				return new IntFloat(id, minDistance);			
			};
		
			// compute the distance to all the explore results
			for (int id : exploreResult) 
				result.add(calcMinDistance.apply(id));
			
			// add all the queries which a present on the current level, 
			// pass the filter but where not in the original list of queries
			final IntSet originalIds = HashIntSets.newImmutableSet(queryImageIds);
			for (int id : queryIds) 
				if(originalIds.contains(id) == false && filter.isValid(id)) 
					result.add(calcMinDistance.apply(id));
		}
		
		return result.stream().mapToInt(IntFloat::getIndex).limit(desiredCount).toArray();
	}
	
	/**
	 * Find a vertex a the given level similar to the image of the given id.
	 * 
	 * @param id
	 * @param atLevel
	 * @param k
	 * @param eps
	 * @return
	 */
	protected int getSimilarVertexAtLevel(int id, int atLevel, int k, float eps) {
		
		// shortcut: the same image is available on the target level
		if(graph.getGraph(atLevel).hasLabel(id))
			return id;
		
		// get feature vector of the id, check the bottom layer which has all images
		final FeatureVector fv = graph.getGraph(0).getFeature(id);
		return graph.searchAtLevel(fv, atLevel, k, eps)[0];
	}
	
	
	// --------------------------------------------------------------------------------------------
	// ---------------------------------- Helper Methods ------------------------------------------
	// --------------------------------------------------------------------------------------------
	
	/**
	 * Erstellt ein 2D array indem belegte Positionen die Flag 255 bekommen.
	 * 
	 * @param localMap
	 * @return array of [y][x]
	 */
	protected static boolean[][] getUsedPositions(GridMap localMap) {
		boolean[][] result = new boolean[localMap.rows()][localMap.columns()];
		for (int i = 0; i < localMap.size(); i++)
			if(localMap.isEmpty(i) == false) // the cells is alreay taken
				result[i / localMap.columns()][i % localMap.columns()] = true;		
		return result;
	}
	

	
	/**
	 * Füllt freie Plätze auf dem Spielfeld mit den Elementen.
	 * 
	 * @param localMap
	 * @param elements
	 * @return number of filled tiles
	 */
	protected static int fillFreePlaces(GridMap localMap, int[] elements) {
		final int listSize = elements.length;
		final int mapSize = localMap.size();
		
		int elementIndex = 0;
		for (int tileIndex = 0; tileIndex < mapSize && elementIndex < listSize; tileIndex++) {
			if(localMap.get(tileIndex) == -1) 
				localMap.set(tileIndex, elements[elementIndex++]);
		}
		return elementIndex;
	}
	
	
	/**
	 * Collect all image-cells which are adjacent to a hole and sort them by their spatial distance to the panelCoordinates
	 * 
	 * @param localMap
	 * @param panelCoordinates
	 * @return
	 */
	protected static IntFloat[] getBorderPositions(GridMap localMap, int[] panelCoordinates) {
		int x1 = panelCoordinates[0]; 
		int y1 = panelCoordinates[1];
		
		int rows = localMap.rows();
		int columns = localMap.columns();
		
		List<IntFloat> borderPositions = new ArrayList<>();		
		for (int y2 = 0; y2 < rows; y2++) {
			for (int x2 = 0; x2 < columns; x2++) {
				
				// ignore holes
				if(localMap.isEmpty(x2, y2)) continue;
				
				// check if an adjacent cell is a hole ...
				boolean isBorder = false;				
				isBorder |= (x2 > 0 && localMap.isEmpty(x2-1, y2)); 			// check left
				isBorder |= (x2 < columns-1 && localMap.isEmpty(x2+1, y2)); 	// check right
				isBorder |= (y2 > 0 && localMap.isEmpty(x2, y2-1)); 			// check top 
				isBorder |= (y2 < rows-1 && localMap.isEmpty(x2, y2+1)); 		// check bottom
				
				// ... remember the current cell
				if(isBorder) {
					int pos = y2 * columns + x2;
					float dist = (float)Math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
					borderPositions.add(new IntFloat(pos, dist));
				}
			}
		}
		
		return borderPositions.stream().sorted(IntFloat.asc()).distinct().toArray(IntFloat[]::new);
	}
	
	/**
	 * Simple wrapper for a integer and float value
	 * 
	 * @author Nico Hezel
	 */
	protected static class IntFloat {
		private final int index;
		private final float distance;
	
		public IntFloat(int index, float distance) {
			this.index = index;
			this.distance = distance;
		}
	
		public int getIndex() {
			return index;
		}
	
		public float getDistance() {
			return distance;
		}
		
		@Override
		public String toString() {
			return "index: "+index+", distance: "+distance;
		}
	
		@Override
		public boolean equals(Object obj) {
			if(obj instanceof IntFloat)
				return ((IntFloat) obj).getIndex() == index;
			return super.equals(obj);
		}
		
		/**
		 * Sort the distances from lowest to highest, followed by the index
		 *
		 * @return
		 */
		public static Comparator<IntFloat> asc() {
			return Comparator.comparingDouble(IntFloat::getDistance).thenComparingInt(IntFloat::getIndex);
		}
	}
}
