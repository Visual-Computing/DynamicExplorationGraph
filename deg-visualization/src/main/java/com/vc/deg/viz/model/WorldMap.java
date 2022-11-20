package com.vc.deg.viz.model;

public class WorldMap extends GridMap {

	public WorldMap(int rows, int columns) {
		super(rows, columns);
	}

	@Override
	public int get(int x, int y) {
		return get(getValidY(y)*columns()+getValidX(x));
	}
	
	@Override
	public void set(int x, int y, int content) {
		set(getValidY(y)*columns()+getValidX(x), content);
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
	
	@Override
	public String toString() {
		return "World columns:"+columns()+", rows:"+rows();
	}
	
	/**
	 * Does not override existing tiles with nulls
	 */
	@Override
	public void copyFrom(GridMap source, int toX, int toY) {
		int sourceRows = source.rows();
		int sourceColumns = source.columns();
		
		for (int y = 0; y < sourceRows; y++) {
			final int thisY = toY + y;
			for (int x = 0; x < sourceColumns; x++) {
				final int content = source.get(x, y);
				final int thisX = toX + x;
				if(content != -1)
					set(thisX, thisY, content);
			}
		}
	}
}
