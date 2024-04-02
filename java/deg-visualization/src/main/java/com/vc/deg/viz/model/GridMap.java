package com.vc.deg.viz.model;

import java.util.Arrays;
import java.util.Random;


/**
 * A grid of ids.
 * All ids must be positive
 * 
 * @author Nico Hezel
 */
public class GridMap {
	
	/**
	 * Create a grid filled with -1 values
	 * 
	 * @param rows must be greater than zero 
	 * @param columns must be greater than zero
	 * @return
	 */
	protected static int[] emptyGrid(int rows, int columns) {
		final int[] cells = new int[rows * columns];
		Arrays.fill(cells, -1);
		return cells;
	}

	// dimensions of the map
	protected final int rows;
	protected final int columns;
	protected final int[] cells;
	
	/**
	 * 
	 * @param rows must be greater than zero 
	 * @param columns must be greater than zero
	 */
	public GridMap(int rows, int columns) {
		this(rows, columns, emptyGrid(rows, columns));
	}
	
	/**
	 * 
	 * @param rows must be greater than zero 
	 * @param columns must be greater than zero
	 * @param cells can not be null
	 */
	public GridMap(int rows, int columns, int[] cells) {
		this.rows = rows;
		this.columns = columns;
		this.cells = cells;
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
	 * Get the content of a cell by x/y
	 * 
	 * @param x
	 * @param y
	 * @return
	 */
	public int get(int x, int y) {
		return get(y*columns()+x);
	}
	
	/**
	 * Get the content of a cell by using the index of the cell.
	 * Indices go from 0 to size()-1.
	 *  
 	 * @param index
	 * @return
	 */
	public int get(int index) {
		return cells[index];
	}
	
	/**
	 * Set the content of a cell
	 * 
	 * @param x
	 * @param y
	 * @param content
	 */
	public void set(int x, int y, int content) {
		set(y*columns()+x, content);
	}
	
	/**
	 * Set the content of a cell using a index.
	 * Indices go from 0 to size()-1.
	 * 
	 * @param index
	 * @param content
	 */
	public void set(int index, int content) {
		cells[index] = content;
	}
	
	/**
	 * Check if the entire map is empty
	 * 
	 * @return
	 */
	public boolean isEmpty() {
		for (int i = 0; i < size(); i++) {
			if(get(i) != -1)
				return false;
		}
		return true;
	}
	
	/**
	 * Check if the cell is empty
	 * 
	 * @param x
	 * @param y
	 * @return
	 */
	public boolean isEmpty(int x, int y) {
		return isEmpty(y*columns()+x);		
	}
	
	/**
	 * Check if the cell is empty
	 * 
	 * @param index
	 * @return
	 */
	public boolean isEmpty(int index) {
		return get(index) == -1;		
	}
	
	
	/**
	 * Count how many map places are empty
	 * 
	 * @return
	 */
	public int freeCount() {
		int freeCount = 0;
		for (int i = 0; i < size(); i++) 
			if(get(i) == -1)
				freeCount++;
		return freeCount;
	}
	
	/**
	 * Shuffle the map cells
	 * 
	 * @param rand
	 */
	public void shuffle(Random rand) {		
		for (int i = 0; i < cells.length; i++) {
			int randomIndexToSwap = rand.nextInt(cells.length);
			int temp = cells[randomIndexToSwap];
			cells[randomIndexToSwap] = cells[i];
			cells[i] = temp;
		}
	}

	/**
	 * Remove the content of the entire map
	 */
	public void clear() {
		Arrays.fill(cells, -1);
	}
	
	/**
	 * Create a copy of this map
	 * @return
	 */
	public GridMap copy() {
		return new GridMap(rows, columns, Arrays.copyOf(cells, cells.length));
	}
	
	@Override
	public String toString() {
		return "Grid with columns:"+columns()+", rows:"+rows();
	}
}
