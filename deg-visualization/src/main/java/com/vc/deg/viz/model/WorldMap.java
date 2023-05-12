package com.vc.deg.viz.model;

import java.util.Arrays;
import java.util.function.IntConsumer;

public class WorldMap {

	// dimensions of the map
	protected final int rows;
	protected final int columns;
	protected final int[] cells;
	
	/** 
	 * @param rows must be greater than zero 
	 * @param columns must be greater than zero
	 */
	public WorldMap(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		this.cells = GridMap.emptyGrid(rows, columns);
	}
	
	public int rows() {
		return rows;
	}

	public int columns() {
		return columns;
	}
	
	/**
	 * Count of all cells in the grid.
	 * Count = Rows * Columns 
	 * 
	 * @return
	 */
	public int size() {
		return rows()*columns();
	}
	
	/**
	 * Get the all none empty cells
	 * 
	 * @param consumer
	 */
	public void foreachCell(IntConsumer consumer) {
		final int size = size();
		for (int i = 0; i < size; i++) {
			final int cell = cells[i];
			if(cell != -1)
				consumer.accept(cell);
		}
	}
	
	
	/**
	 * Remove the content of the entire map
	 */
	public void clear() {
		Arrays.fill(cells, -1);
	}
	
	/**
	 * Get a valid x position. Wrap around if necessary.
	 * 
	 * @param x
	 * @return
	 */
	public int getValidX(int x) {
		while(x < 0) 
			x = columns() + x;
		while(x >= columns())
			x = x - columns();
		return x;
	}
	
	/**
	 * Get a valid y position. Wrap around if necessary.
	 * 
	 * @param y
	 * @return
	 */
	public int getValidY(int y) {
		while(y < 0) 
			y = rows() + y;
		while(y >= rows())
			y = y - rows();
		return y;
	}
	
	/**
	 * Copy tile information from the given GridMap to this GridMap at a specific offset.
	 *  
	 * @param source
	 * @param toX
	 * @param toY
	 */
	public void copyFrom(GridMap source, int toX, int toY) {
		final int sourceRows = source.rows();
		final int sourceColumns = source.columns();
		
		for (int y = 0; y < sourceRows; y++) {
			final int thisY = getValidY(toY + y);
			for (int x = 0; x < sourceColumns; x++) {
				final int content = source.get(x, y);
				final int thisX = getValidX(toX + x);
				if(content != -1)
					cells[thisY*columns()+thisX] = content;
			}
		}
	}
	
	/**
	 * Copy tile information from this GridMap at a specific offset to the given GridMap. 
	 * 
	 * @param destination
	 * @param fromX
	 * @param fromY
	 */
	public void copyTo(GridMap destination, int fromX, int fromY) {
		final int destinationRows = destination.rows();
		final int destinationColumns = destination.columns();
		
		for (int y = 0; y < destinationRows; y++) {
			final int thisY = getValidY(fromY + y);
			for (int x = 0; x < destinationColumns; x++) {				
				final int thisX = getValidX(fromX + x);
				final int cell = cells[thisY*columns()+thisX];
				destination.set(x, y, cell);
			}
		}	
	}
	
	@Override
	public String toString() {
		return "World columns:"+columns()+", rows:"+rows();
	}
}
