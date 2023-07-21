package com.vc.deg.viz.model;

public class MotionVector {

	public static final MotionVector UP = new MotionVector(0, -1);
	public static final MotionVector DOWN = new MotionVector(0, 1);
	public static final MotionVector LEFT = new MotionVector(-1, 0);
	public static final MotionVector RIGHT = new MotionVector(1, 0);
	
	protected final double x;
	protected final double y;
	
	public MotionVector(double x, double y) {
		this.x = x;
		this.y = y;
	}
	
	public double getX() {
		return x;
	}
	
	public double getY() {
		return y;
	}	
	
	public double length() {
		return Math.abs(x) + Math.abs(y);
	}
	
	public MotionVector normalize() {
		double max = Math.max(Math.abs(x), Math.abs(y));
		return new MotionVector(x / max, y / max);
	}
	
	@Override
	public String toString() {
		return "x:"+x+", y:"+y;
	}
}
