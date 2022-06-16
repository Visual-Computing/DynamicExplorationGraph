package com.vc.deg.navigation;

import com.vc.deg.graph.EvenRegularWeightedUndirectedGraph;

/**
 * Erzeugt 2D projektionen von Subgraphen und übernimmt die Navigation davon.
 * Stellt allgemeine Suchfunktionen bereits
 * 
 * @author Neiko
 *
 */
public class EvenRegularGraphNavigator implements GraphNavigator {

	protected EvenRegularWeightedUndirectedGraph graph;

	public EvenRegularGraphNavigator(EvenRegularWeightedUndirectedGraph graph) {
		this.graph = graph;
	}

}
