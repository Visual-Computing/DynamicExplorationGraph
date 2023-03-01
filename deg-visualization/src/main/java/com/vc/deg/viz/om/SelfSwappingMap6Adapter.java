package com.vc.deg.viz.om;

import java.util.function.IntFunction;

import com.vc.deg.viz.model.GridMap;
import com.vc.deg.viz.om.SelfSwappingMap6.MapPlace;


/** 
 * Can handle holes
 * 
 * @author barthel and hezel
 */
public class SelfSwappingMap6Adapter {

	private IntFunction<float[]> idToFloatFeature;

	public SelfSwappingMap6Adapter(IntFunction<float[]> idToFloatFeature) {
		this.idToFloatFeature = idToFloatFeature;
	}
	
	/**
	 * Start the swapping process
	 * 
	 * @param map
	 * @param inUse[y][x]
	 */
	public void arrangeWithHoles(GridMap map, short[][] inUse)  {
		int rows = map.rows();
		int columns = map.columns();
				
		// copy the data
		MapPlace[] mapPlaces = new MapPlace[columns * rows];
		for (int y = 0; y < rows; y++) {		
			for (int x = 0; x < columns; x++) {
				int content = map.get(x, y);
				if(content != -1) {
					float[] floatFeature = idToFloatFeature.apply(content);
					mapPlaces[x + y * columns] = new MapPlace(content, floatFeature, inUse[y][x] == 0);
				}
			}
		}
		
		SelfSwappingMap6 ssm = new SelfSwappingMap6(mapPlaces, columns, rows, 7);
		ssm.setWrap(false);
		ssm.run();
		
		// apply the new order to the map
		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < columns; x++) {
				MapPlace mapPlace = mapPlaces[x + y * columns];
				if(mapPlace != null) {
					map.set(x, y, mapPlace.getId());
				} else {
					map.set(x, y, -1);
				}
			}
		}
	}
	
}
