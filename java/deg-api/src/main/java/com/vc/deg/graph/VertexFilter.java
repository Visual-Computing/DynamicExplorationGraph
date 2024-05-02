package com.vc.deg.graph;

import java.util.function.IntConsumer;

/**
 * A filter to check labels used in a search or exploration task
 * 
 * @author Nico Hezel
 */
public interface VertexFilter {

	/**
	 * Is the label valid
	 * 
	 * @param label
	 * @return
	 */
	public boolean isValid(int label);
	
	/**
	 * Number of valid labels
	 * 
	 * @return
	 */
	public int size();
	
	/**
	 * Include rate
	 * 
	 * @return
	 */
	public float getInclusionRate();
	
	/**
	 * For each valid id in the filter the action called
	 * 
	 * @param action
	 */
	public void forEachValidId(IntConsumer action);	
}