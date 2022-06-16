package com.vc.deg.navigation;

import com.vc.deg.data.DataRepository;
import com.vc.deg.graph.EvenRegularWeightedUndirectedGraph;
import com.vc.deg.navigation.GraphNavigator;

/**
 * Erzeugt 2D projektionen von Subgraphen und übernimmt die Navigation davon.
 * Stellt allgemeine Suchfunktionen bereits
 * 
 * @author Neiko
 *
 */
public class EvenRegularGraphNavigator<T> implements GraphNavigator {

	protected EvenRegularWeightedUndirectedGraph graph;
	protected DataRepository<T> repository;

	public EvenRegularGraphNavigator(EvenRegularWeightedUndirectedGraph graph, DataRepository<T> repository) {
		this.graph = graph;
		this.repository = repository;
	}

}
