package com.vc.deg.ref.navigation;

import com.vc.deg.GraphNavigator;
import com.vc.deg.graph.WeightedUndirectedGraph;

/**
 * Erzeugt 2D projektionen von Subgraphen und übernimmt die Navigation davon.
 * Stellt allgemeine Suchfunktionen bereits
 * 
 * @author Neiko
 *
 */
public class EvenRegularGraphNavigator implements GraphNavigator {

	protected WeightedUndirectedGraph graph;

	public EvenRegularGraphNavigator(WeightedUndirectedGraph graph) {
		this.graph = graph;
	}

}
