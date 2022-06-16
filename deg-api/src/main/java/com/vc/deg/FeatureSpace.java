package com.vc.deg;

/**
 * Eine Klasse die Features vergleichen kann und generell we
 * 
 * @author Nico Hezel
 */
public interface FeatureSpace {

	/**
	 * Size in bytes per feature vector
	 * 
	 * @return
	 */
	public int size();
	
	/**
	 * This method does not check the size of the memory view
	 * 
	 * @param f1
	 * @param f2
	 * @return
	 */
	public float computeDistance(MemoryView f1, MemoryView f2);
}
