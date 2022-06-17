package com.vc.deg;

/**
 * Eine Klasse die Features vergleichen kann und generell we
 * The feature space knows how the data in the MemoryView is structured
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
	public float computeDistance(FeatureVector f1, FeatureVector f2);
}
