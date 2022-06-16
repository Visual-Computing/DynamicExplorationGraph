package com.vc.deg.data;

/**
 * Eine Klasse die Features vergleichen kann und generell we
 * 
 * @author Neiko
 *
 * @param <T>
 */
public interface FeatureSpace<T> {

	public float computeDistance(T f1, T f2);
	
}
