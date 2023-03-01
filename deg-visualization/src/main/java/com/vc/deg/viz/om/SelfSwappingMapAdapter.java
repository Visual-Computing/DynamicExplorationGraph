package com.vc.deg.viz.om;

import java.util.function.IntFunction;

import com.vc.deg.viz.model.GridMap;
import com.vc.deg.viz.om.SelfSwappingMap2.MapPlace;


/** 
 * Can not handle holes 
 * 
 * @author barthel and hezel
 */
public class SelfSwappingMapAdapter {

	private final IntFunction<float[]> idToFloatFeature;

	public SelfSwappingMapAdapter(IntFunction<float[]> idToFloatFeature) {
		this.idToFloatFeature = idToFloatFeature;
	}
	
	
	/**
	 * Start the swapping process
	 * 
	 * @param map
	 * @param inUse[y][x]
	 */
	public void arrange(GridMap map, short[][] inUse)  {

		// copy the data
		MapPlace[][] mapPlaces = new MapPlace[map.rows()][map.columns()];
		for (int y = 0; y < map.rows(); y++) {
			for (int x = 0; x < map.columns(); x++) {
				int content = map.get(x, y);
				if(content != -1) {
					float[] floatFeature = idToFloatFeature.apply(content);
					mapPlaces[y][x] = new MapPlace(content, floatFeature);
				}
			}
		}
		
		SelfSwappingMap2 ssm = new SelfSwappingMap2(10, 10);
		ssm.run(mapPlaces, inUse);
		
		// apply the new order to the map
		for (int y = 0; y < map.rows(); y++) {
			for (int x = 0; x < map.columns(); x++) {
				MapPlace mapPlace = mapPlaces[y][x];
				if(mapPlace != null) {
					map.set(x, y, mapPlace.getId());
				}
			}
		}
	}
}
