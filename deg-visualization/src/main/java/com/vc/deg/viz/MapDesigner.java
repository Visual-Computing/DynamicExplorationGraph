package com.vc.deg.viz;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
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
	 * @param filter can be null
	 * @param worldMap
	 * @param atLevel
	 * @return
	 */
	protected GraphFilter prepareFilter(MutableGraphFilter filter, WorldMap worldMap, int atLevel) {
		if(filter == null) {
			final DynamicExplorationGraph deg = graph.getGraph(atLevel);		
			final IntSet validIds = HashIntSets.newMutableSet(c -> {
				deg.forEachVertex((id, fv) -> c.accept(id));			
			}, deg.size());
			
			for (int i = 0; i < worldMap.size(); i++) {
				final int cell = worldMap.get(i);
				if(cell != -1)
					validIds.removeInt(cell);
			}
			return new PreparedGraphFilter(validIds);
		}
		
		return filter.remove(removeFunc -> {
			for (int i = 0; i < worldMap.size(); i++) {
				final int cell = worldMap.get(i);
				if(cell != -1)
					removeFunc.accept(cell);
			}
		});
	}
	
	/**
	 * Fill the map with the elements from the world map. If there are still holes fill them with the neighboring elements.
	 * In the case there are no elements at all an empty map will be returned.
	 * 
	 * @param localMap
	 * @param worldPosX (world coordinates)
	 * @param worldPosY (world coordinates)
	 * @param atLevel
	 * @param filter can be null
	 */
	public void fill(WorldMap worldMap, GridMap localMap, int worldPosX, int worldPosY, int atLevel, MutableGraphFilter filter) {
		
		// kein Graph
		if(graph == null) 
			return;
		
		// lade existierende Panels auf die lokale Karte
		worldMap.copyTo(localMap, worldPosX, worldPosY);
		
		// finde Nachbarn im Graph die noch nicht auf der Karte sind 
		int emptyTileCount = localMap.freeCount();
		if(emptyTileCount > 0 && localMap.isEmpty() == false) {
			
			// welche Plätze[y][x] sind schon belegt
			boolean[][] inUse = getUsedPositions(localMap);
			
			// get all border images ignore there order
			IntFloat[] borderPanelPos = getBorderPositions(localMap, new int[] {0,0});
			int[] queryImages = Arrays.stream(borderPanelPos).mapToInt(p -> localMap.get(p.getIndex())).toArray();
			
			// Besorge die Nachbarn der Randbilder
			long start = System.currentTimeMillis();
			int[] images = getBestNeighborNodes(queryImages, atLevel, emptyTileCount, prepareFilter(filter, worldMap, atLevel));
			
			// fülle die fehlenden Felder mit den Nachbarn aus
			int fillCount = fillFreePlaces(localMap, images);
			
			// es sind neue element hinzu gekommen
			if(fillCount > 0) {
				log.debug("Fill free tiles ("+images.length+" elements) took "+(System.currentTimeMillis() - start)+"ms");
				
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
	 * Jump in the graph to a position most similar to the content. Place the content at the target position in the map.
	 * Arrange other similar images from the graph around the image on the map. Use the graph at given the level. 
	 * 
	 * @param worldMap
	 * @param localMap
	 * @param selectedContent
	 * @param targetPosX (local coordinates)
	 * @param targetPosY (local coordinates)
	 * @param worldPosX (world coordinates)
	 * @param worldPosY (world coordinates)
	 * @param atLevel
	 * @param filter can be null
	 */
	public void jump(WorldMap worldMap, GridMap localMap, int selectedContent, int targetPosX, int targetPosY, int worldPosX, int worldPosY, int atLevel, MutableGraphFilter filter) {
		
		// kein Graph
		if(graph == null) 
			return;
		
		// besorge die Nachbarschaft von dem selektierten Bild
		final long start = System.currentTimeMillis();
		int[] elements = getBestNeighborNodes(new int[] {selectedContent}, atLevel, localMap.size(), prepareFilter(filter, worldMap, atLevel));
		log.debug("Collecting neighbourhood ("+elements.length+" elements) took "+(System.currentTimeMillis() - start)+"ms");
		
		// sortiere die Bilder
		arrange(worldMap, localMap, selectedContent, targetPosX, targetPosY, worldPosX, worldPosY, elements);
	}
	
	/**
	 * Arranges the elements on the local map.
	 * Stores the arrangement in the world map at the given position.
	 * The provided elements do not have to be in the graph.
	 * Place the selectedContent at the target position in the map.
	 * 
	 * @param worldMap
	 * @param localMap
	 * @param selectedContent
	 * @param targetPosX
	 * @param targetPosY
	 * @param worldPosX
	 * @param worldPosY
	 * @param elements
	 */
	protected void arrange(WorldMap worldMap, GridMap localMap, int selectedContent, int targetPosX, int targetPosY, int worldPosX, int worldPosY, int[] elements) {
		final long start = System.currentTimeMillis();
		
		// platziere die Bilder auf der Karte 
		localMap.clear();
		for (int i = 0; i < elements.length; i++) 
			localMap.set(i, elements[i]);
		
		// sorge dafür dass das selektierte Bild in der gewünschten Position ist		
		int targetTile = localMap.get(targetPosX, targetPosY);
		localMap.set(localMap.columns() - 1 , localMap.rows() - 1, targetTile);
		localMap.set(targetPosX, targetPosY, selectedContent);
				
		// blockiere die gewünschten Position
		boolean[][] blockedPanels = new boolean[localMap.rows()][localMap.columns()];
		blockedPanels[targetPosY][targetPosX] = true;
		
		// sortiere die Bilder
		arrangeMap(localMap, blockedPanels);
		
		// die Sortierung auf die Weltkarte kopieren
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
	 * @param filter can be null
	 */
	public void move(WorldMap worldMap, GridMap localMap, MotionVector shiftVector, MotionVector directionVector, int worldPosX, int worldPosY, int atLevel, MutableGraphFilter filter) {
		
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
			int[] images = getNeighbors(localMap, shiftVector, directionVector, atLevel, prepareFilter(filter, worldMap, atLevel));
			
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
	 * Besorge Nachbar Bilder aus dem Graph von all den Bildern auf die der ShiftVector zeigt oder die in der Nähe sind.
	 * 
	 * 
	 * @param localMap
	 * @param shiftVector
	 * @param directionVector
	 * @param atLevel
	 * @param filter
	 * @return number of new elements
	 */
	protected int[] getNeighbors(GridMap localMap, MotionVector shiftVector, MotionVector directionVector, int atLevel, GraphFilter filter) {		
		final float rows = localMap.rows();
		final float columns = localMap.columns();
		
		// extend the direction vector until it hits the a tile at the border
		final MotionVector nDirection = directionVector.normalize();
		final float cdx = (int)(nDirection.getX() * (columns/2f - Math.abs(shiftVector.getX())));
		final float cdy = (int)(nDirection.getY() * (rows/2f - Math.abs(shiftVector.getY())));
		
		// das durch den Richtungs-Vector getroffene Tile im Viewport
		int arrayPosX = Math.round(columns/2f - cdx); 
		int arrayPosY = Math.round(rows/2f    - cdy);
		if(shiftVector.getY() > 0) arrayPosY--;
		if(shiftVector.getX() > 0) arrayPosX--;
		final int[] panelCoordinates = new int[] {arrayPosX, arrayPosY};
						
		// besorge Bilder entlang der Viewport Border, in der Nähe des Bewegungsvektors 
		IntFloat[] borderPanelPos = getBorderPositions(localMap, panelCoordinates);
		borderPanelPos = Arrays.copyOf(borderPanelPos, Math.max(1, borderPanelPos.length/2)); // halbiere dessen Anzahl
		int[] queryImages = Arrays.stream(borderPanelPos).mapToInt(p -> localMap.get(p.getIndex())).toArray();
		
		// finde Nachbarn im Graph die noch nicht auf der Karte sind 
		final int desiredCount = localMap.freeCount();	
		return getBestNeighborNodesMedian(queryImages, atLevel, desiredCount, filter);
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
	 * Find the three median images from the list of query images and find their best neighbors with {@link #getBestNeighborNodes(int[], int, int, GraphFilter)}}
	 * 
	 * @param queryImages
	 * @param atLevel
	 * @param desiredCount
	 * @param filter
	 * @return
	 */
	public int[] getBestNeighborNodesMedian(int[] queryImages, int atLevel, int desiredCount, GraphFilter filter) {
		
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
		return getBestNeighborNodes(medianImages, atLevel, desiredCount, filter);
	}
	
	/**
	 * Finde alle Nachbarn von den Bildern in der Anfrage. Verwende aber nur die 
	 * Nachbarn die besonders ähnlich zu eines der Query Bilder sind (max similarity)
	 * und wo der angegebene Filter true zurück liefert.
	 * 	 
	 * Query Bilder sind nicht in der Ergebnismenge dieser Methode enthalten.
	 * Besorge so viele Bilder bis desiredCount erreicht wurde.
	 * Die Reihenfolge geht von den wichtigen zu den unwichtigsten.
	 * 
	 * Wobei nur Bilder betrachtet werde die durch den Filter kommen.
	 * 
	 * @param desiredQueryImages
	 * @param atLevel
	 * @param desiredCount
	 * @param filter
	 * @return
	 */
	protected int[] getBestNeighborNodes(int[] desiredQueryImages, int atLevel, int desiredCount, GraphFilter filter) {
		
		// not enough images or invalid goal
		if(desiredCount <= 0 || desiredQueryImages.length == 0) 
			return new int[0];
		
		// search for similar query images at the given level, for all query images which are not present at the level
		final int[] queryIds = HashIntSets.newImmutableSet(c -> {			
			for (int desiredQueryImage : desiredQueryImages) 
				c.accept(getSimilarNodeAtLevel(desiredQueryImage, atLevel, 1, 0.6f));
		}).toIntArray();
		
		// explore from the query ids other vertices in the graph which are valid in the filter
		return graph.exploreAtLevel(queryIds, atLevel, desiredCount, Integer.MAX_VALUE, filter);
	}
	
	protected int getSimilarNodeAtLevel(int id, int atLevel, int k, float eps) {
		
		// shortcut: ist das selbe Bild auf dem Level
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
			if(localMap.isEmpty(i) == false) // ist das Feld bereits belegt
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
	 * Liefert alle Bilder deren Nachbarn eine Lücke sind.
	 * Und sortiert diese Anhand ihrer nähe zu den panelCoordinates.
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
				
				// ignoriere Lücken
				if(localMap.isEmpty(x2, y2)) continue;
				
				// ist einer der Vier Himmelsrichtungen eine Lücke ...
				boolean isBorder = false;				
				isBorder |= (x2 > 0 && localMap.isEmpty(x2-1, y2)); 			// check left
				isBorder |= (x2 < columns-1 && localMap.isEmpty(x2+1, y2)); 	// check right
				isBorder |= (y2 > 0 && localMap.isEmpty(x2, y2-1)); 			// check top 
				isBorder |= (y2 < rows-1 && localMap.isEmpty(x2, y2+1)); 		// check bottom
				
				// ... dann merke die aktuelle Position
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
