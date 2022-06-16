package com.vc.deg.data;

import com.vc.deg.data.FeatureSpace;

/**
 * Hier werden die Metadaten zu den Knoten im Graphen gehalten
 * 
 * @author Neiko
 *
 */
public interface DataRepository<T> {

	public FeatureSpace<T> getFeatureSpace();	
	public void add(int id, T obj);
	public T get(int id);
}
