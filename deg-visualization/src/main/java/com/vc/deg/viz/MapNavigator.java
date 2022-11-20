package com.vc.deg.viz;

import com.vc.deg.viz.filter.MutableGraphFilter;
import com.vc.deg.viz.model.GridMap;
import com.vc.deg.viz.model.MotionVector;
import com.vc.deg.viz.model.WorldMap;

/**
 * The MapNavigator is statefull in regards to the user calling it
 * 
 * @author Nico Hezel
 */
public class MapNavigator  {

	protected final WorldMap worldMap;
	protected final MapDesigner designer;
		
	// current level of the navigator
	protected int currentLevel;
	
	// current position on the world map, where the local map starts
	protected int worldPosX;
	protected int worldPosY;
	
	public MapNavigator(WorldMap worldMap, MapDesigner mapDesigner) {
		this.worldMap = worldMap;
		this.designer = mapDesigner;
		
		// start in the middle of the world
		reset();
	}

	public int getLevel() {
		return currentLevel;
	}
	
	public int getWorldPosX() {
		return worldPosX;
	}
	
	public int getWorldPosY() {
		return worldPosY;
	}
	
	@Override
	public String toString() {
		return "Position "+worldPosX+"/"+worldPosY;
	}
	
	// -------------------------------------------------------------------------------------------
	// ----------------------------- Navigation Methods ------------------------------------------
	// -------------------------------------------------------------------------------------------
	
	/**
	 * Set the current world map position
	 * 
	 * @param x
	 * @param y
	 */
	protected void setWorldPosition(int x, int y) {
		this.worldPosX = worldMap.getValidX(x);
		this.worldPosY = worldMap.getValidY(y);
	}
	
	/**
	 * Reset the entire world map and the position of the navigator
	 */
	public void reset() {
		this.worldPosX = worldMap.columns() / 2;
		this.worldPosY = worldMap.rows() / 2;
		this.worldMap.clear();
	}
		
	/**
	 * Move x/y steps on the world map and fill the local map with the resulting world map data.
	 * A normalized direction vector (sum of components = 1) allows fine-grain definition of the direction.
	 * 
	 * @param map
	 * @param x
	 * @param y
	 * @param directionX
	 * @param directionY
	 * @param filter can be null
	 * @return
	 */
	public boolean move(GridMap map, int x, int y, double directionX, double directionY, MutableGraphFilter filter) {
		
		if(x != 0 || y != 0) {
			setWorldPosition(worldPosX - x, worldPosY - y);
			
			// Weltkarten aktualisieren		
			MotionVector shiftVector = new MotionVector(x, y);
			MotionVector directionVector = new MotionVector(directionX, directionY);
			designer.move(worldMap, map, shiftVector, directionVector, worldPosX, worldPosY, this.currentLevel, filter);
			
			return true;
		}

		return false;
	}
	
	/**
	 * Clear the world map and reset the position of the navigator to the center of the now empty world.
	 * Jump in the graph to a position most similar to the content. Place the content at the target position in the map.
	 * Arrange other similar images from the graph around the image on the map. Use the graph at given the level. 
	 * 
	 * @param map
	 * @param content
	 * @param posX
	 * @param posY
	 * @param toLevel
	 * @param filter can be null
	 */
	public void jump(GridMap map, int content, int posX, int posY, int toLevel, MutableGraphFilter filter) {	
		this.currentLevel = toLevel;
		reset();
		designer.jump(worldMap, map, content, posX, posY, worldPosX, worldPosY, toLevel, filter);
	}

	/**
	 * Move to the position on the world map. Fill the local map with the elements from the world map. 
	 * If there are still holes fill them with the neighboring elements.
	 * In the case there are no elements at all, an empty map will be returned.
	 *  
	 * @param map
	 * @param worldPosX
	 * @param worldPosY
	 * @param filter can be null
	 */
	public void explore(GridMap map, int worldPosX, int worldPosY, MutableGraphFilter filter) {
		setWorldPosition(worldPosX, worldPosY);
		designer.fill(worldMap, map, worldPosX, worldPosY, this.currentLevel, filter);
	}

}