package com.vc.deg.viz.om;

import java.util.Random;
import java.util.function.IntFunction;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.viz.model.GridMap;
import com.vc.deg.viz.om.FLASnoMapSorter.MapPlace;


/** 
 * Can handle holes and fixed cells
 * 
 * @author barthel and hezel
 */
public class FLASnoMapSorterAdapter {

	private final IntFunction<FeatureVector> idToFloatFeature;
	private final FeatureSpace distFunc;

	public FLASnoMapSorterAdapter(IntFunction<FeatureVector> idToFloatFeature, FeatureSpace distFunc) {
		this.idToFloatFeature = idToFloatFeature;
		this.distFunc = distFunc;
	}
	
	/**
	 * Start the swapping process
	 * 
	 * @param map
	 * @param inUse[y][x]
	 */
	public void arrangeWithHoles(GridMap map, boolean[][] inUse)  {
		final int rows = map.rows();
		final int columns = map.columns();
				
		// copy the data
		final MapPlace[] mapPlaces = new MapPlace[columns * rows];
		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < columns; x++) {
				final int id = map.get(x, y);
				final FeatureVector feature = (id == -1) ? null : idToFloatFeature.apply(id);
				mapPlaces[x + y * columns] = new MapPlace(id, feature, inUse[y][x]);
			}
		}
		
		final Random rnd = new Random(7);
		final FLASnoMapSorter flas = new FLASnoMapSorter(rnd);
		flas.doSorting(mapPlaces, columns, rows, distFunc);
		
		// apply the new order to the map
		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < columns; x++) {
				final MapPlace mapPlace = mapPlaces[x + y * columns];
				map.set(x, y, mapPlace.getId());
			}
		}
	}	
}