package com.vc.deg.viz.model;

import java.util.function.IntConsumer;

import com.koloboke.collect.map.LongObjMap;
import com.koloboke.collect.map.hash.HashLongObjMaps;

/**
 * The dynamic world map stores small region (e.g. the viewport).
 * For good performance the size of the internal regions should be the same as the viewport.
 * 
 * @author Nico Hezel
 */
public class WorldMap {

	protected final LongObjMap<GridMap> maps;
	protected final int regionRows;
	protected final int regionColumns;
	
	/**
	 * The world map stores small region (e.g. the viewport) onto a bigger dynamic map.
	 * 
	 * @param regionRows internal width of the region 
	 * @param regionColumns internal height of a the region
	 */
	public WorldMap(int regionRows, int regionColumns) {
		this.maps = HashLongObjMaps.newMutableMap();
		this.regionColumns = regionColumns;
		this.regionRows = regionRows;
	}
	
	
	/**
	 * Get the all none empty cells
	 * 
	 * @param consumer
	 */
	public void foreachCell(IntConsumer consumer) {
		for (GridMap map : maps.values()) {
			final int size = map.size();
			for (int i = 0; i < size; i++) {
				final int cell = map.get(i);
				if(cell != -1)
					consumer.accept(cell);
			}
		}
	}
	
	/**
	 * Remove the content of the entire map
	 */
	public void clear() {
		maps.clear();
	}
	
	/**
	 * Get a region from the global map
	 * 
	 * @param x position of the region 
	 * @param y position of the region 
	 * @return
	 */
	private GridMap getOrCreate(int x, int y) {
		final long key = (((long)x) << 32) | (y & 0xffffffffL);
		return maps.computeIfAbsent(key, k -> {
			return new GridMap(regionRows, regionColumns);			
		});
	}
	
	/**
	 * Modulo for positive and negative numbers of x.
	 * Assuming N > 0 and N + N - 1 <= INT_MAX
	 * 
	 * @param x
	 * @param N
	 * @return
	 */
	private static int modulo(int x, int N){
	    return (x % N + N) % N;
	}
	
	/**
	 * Copy information from the given GridMap to the WorldMap at a specific offset.
	 *  
	 * @param source
	 * @param toX (can be negative)
	 * @param toY (can be negative)
	 */
	public void copyFrom(GridMap source, int toX, int toY) {
		final int sourceRows = source.rows();
		final int sourceColumns = source.columns();
		
		// iterate over all regions which contain parts of the required data
		final int regionStartY = toY - modulo(toY, regionRows);
		final int regionStartX = toX - modulo(toX, regionColumns);
		for (int regionY = regionStartY; regionY < toY + sourceRows; regionY+=regionRows) {			
			for (int regionX = regionStartX; regionX < toX + sourceColumns; regionX+=regionColumns) {
				final GridMap target = getOrCreate(regionX, regionY);
				
				// iterate over all cells in the grid map which need to be transfered
				final int targetStartY = Math.max(0, toY - regionY);
				final int targetStartX = Math.max(0, toX - regionX);
				final int targetEndY = regionRows - Math.max(0, (regionY + regionRows) - (toY + sourceRows));
				final int targetEndX = regionColumns - Math.max(0, (regionX + regionColumns) - (toX + sourceColumns));
				for (int targetY = targetStartY; targetY < targetEndY; targetY++) {
					for (int targetX = targetStartX; targetX < targetEndX; targetX++) {
						final int sourceX = (targetX + regionX) - toX;
						final int sourceY = (targetY + regionY) - toY;
						final int content = source.get(sourceX, sourceY);
						
						// copy only positive content to the region map
						if(content != -1) 
							target.set(targetX, targetY, content);
					}
				}
			}
		}
	}
	
	/**
	 * Copy information from the WorldMap at a specific offset to the given GridMap. 
	 * 
	 * @param destination
	 * @param fromX (can be negative)
	 * @param fromY (can be negative)
	 */
	public void copyTo(GridMap destination, int fromX, int fromY) {
		final int destinationRows = destination.rows();
		final int destinationColumns = destination.columns();
		
		
		// iterate over all regions which contain parts of the required data
		final int regionStartY = fromY - modulo(fromY, regionRows);
		final int regionStartX = fromX - modulo(fromX, regionColumns);
		for (int regionY = regionStartY; regionY < fromY + destinationRows; regionY+=regionRows) {			
			for (int regionX = regionStartX; regionX < fromX + destinationColumns; regionX+=regionColumns) {
				final GridMap target = getOrCreate(regionX, regionY);
				
				// iterate over all cells in the grid map which need to be transfered
				final int targetStartY = Math.max(0, fromY - regionY);
				final int targetStartX = Math.max(0, fromX - regionX);
				final int targetEndY = regionRows - Math.max(0, (regionY + regionRows) - (fromY + destinationRows));
				final int targetEndX = regionColumns - Math.max(0, (regionX + regionColumns) - (fromX + destinationColumns));
				for (int targetY = targetStartY; targetY < targetEndY; targetY++) {
					for (int targetX = targetStartX; targetX < targetEndX; targetX++) {
						final int content = target.get(targetX, targetY);
						
						// copy content to the destination map
						final int sourceX = (targetX + regionX) - fromX;
						final int sourceY = (targetY + regionY) - fromY;
						destination.set(sourceX, sourceY, content);
					}
				}
			}
		}	
	}
	
	@Override
	public String toString() {
		return "Dynamic World Map with "+maps.size()+" regions";
	}
}
