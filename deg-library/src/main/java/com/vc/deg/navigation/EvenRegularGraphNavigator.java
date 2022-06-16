package com.vc.deg.navigation;

import com.vc.deg.data.DataComparator;
import com.vc.deg.graph.EvenRegularWeightedUndirectedGraph;
import com.vc.deg.navigation.GraphNavigator;

/**
 * Erzeugt 2D projektionen von Subgraphen und übernimmt die Navigation davon.
 * Stellt allgemeine Suchfunktionen bereits
 * 
 * @author Neiko
 *
 */
public class EvenRegularGraphNavigator implements GraphNavigator {

	protected EvenRegularWeightedUndirectedGraph graph;
	protected DataComparator comparator;

	public EvenRegularGraphNavigator(EvenRegularWeightedUndirectedGraph graph, DataComparator comparator) {
		this.graph = graph;
		this.comparator = comparator;
	}

}
