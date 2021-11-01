/*-
 * #%L
 * This file is part of QuPath.
 * %%
 * Copyright (C) 2014 - 2016 The Queen's University of Belfast, Northern Ireland
 * Contact: IP Management (ipmanagement@qub.ac.uk)
 * Copyright (C) 2018 - 2020 QuPath developers, The University of Edinburgh
 * %%
 * QuPath is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * QuPath is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License 
 * along with QuPath.  If not, see <https://www.gnu.org/licenses/>.
 * #L%
 */

package qupath.lib.roi;

import org.locationtech.jts.geom.Geometry;

import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.GeometryTools.GeometryConverter;
import qupath.lib.roi.interfaces.ROI;

/**
 * Abstract implementation of a ROI.
 * 
 * @author Pete Bankhead
 *
 */
abstract class AbstractPathROI implements ROI {
	
	// Dimension variables
	int c = -1; // Defaults to -1, indicating all channels
	int t = 0;  // Defaults to 0, indicating first time point
	int z = 0;  // Defaults to 0, indicating first z-slice
	
	private transient ImagePlane plane;
	
	public AbstractPathROI() {
		this(null);
	}
	
	public AbstractPathROI(ImagePlane plane) {
		super();
		if (plane == null)
			plane = ImagePlane.getDefaultPlane();
		this.c = plane.getC();
		this.z = plane.getZ();
		this.t = plane.getT();
	}
	
	@Override
	public ImagePlane getImagePlane() {
		if (plane == null)
			plane = ImagePlane.getPlaneWithChannel(c, z, t);
		return plane;
	}
	
//	Object asType(Class<?> cls) {
//		if (cls.isInstance(this))
//			return this;
//		
//		if (cls == Geometry.class)
//			return ConverterJTS.getGeometry(this);
//
//		if (cls == Shape.class)
//			return PathROIToolsAwt.getShape(this);
//		
//		throw new ClassCastException("Cannot convert " + t + " to " + cls);
//	}
	
	@Override
	public int getZ() {
		return z;
	}
	
	@Override
	public int getT() {
		return t;
	}

	@Override
	public int getC() {
		return c;
	}
	
	/**
	 * True if the bounding box has zero area
	 */
	@Override
	public boolean isEmpty() {
		int n = getNumPoints();
		if (n == 0)
			return true;
		if (isArea())
			return getArea() == 0;
		if (isLine())
			return getLength() == 0;
		return false;
	}
	
	@Override
	public String toString() {
		var sb = new StringBuilder(getRoiName())
			.append(" (")
			.append(Math.round(getBoundsX()))
			.append(", ")
			.append(Math.round(getBoundsY()))
			.append(", ")
			.append(Math.round(getBoundsWidth()))
			.append(", ")
			.append(Math.round(getBoundsHeight()));
		if (getZ() != 0)
			sb.append(", z=").append(getZ());
		if (getT() != 0)
			sb.append(", t=").append(getT());
		if (getC() != -1)
			sb.append(", c=").append(getC());
		sb.append(")");
		return sb.toString();
		
//		Rectangle bounds = getBounds();
//		return String.format("%s (%.0f, %.0f, %.0f, %.0f, z=%d, t=%d, c=%d)",
//				getRoiName(),
//				getBoundsX(), getBoundsY(), getBoundsWidth(), getBoundsHeight(),
//				getZ(), getT(), getC());
//		String name = getName();
//		if (name != null)
////			return name;			
//			return name + " - " + getROIType();			
//		return getRoiName();
//		return String.format("%s (%.1f, %.1f)", getROIType(), getCentroidX(), getCentroidY());
//		return "Me";
	}
	
	private transient int nPoints =  -1;
	
	/**
	 * Default implementation, calls {@link #getAllPoints()} and then caches the result.
	 * Subclasses may override for efficiency.
	 * @return
	 * @implNote the default implementation assumes that ROIs are immutable.
	 */
	@Override
	public int getNumPoints() {
		if (nPoints < 0)
			nPoints = getAllPoints().size();
		return nPoints;
	}

	@Override
	public boolean isLine() {
		return getRoiType() == RoiType.LINE;
	}
	
	@Override
	public boolean isArea() {
		return getRoiType() == RoiType.AREA;
	}
	
	@Override
	public boolean isPoint() {
		return getRoiType() == RoiType.POINT;
	}
	
	private static GeometryConverter converter = new GeometryConverter.Builder().build();
	
	@Override
	public Geometry getGeometry() {
		return converter.roiToGeometry(this);
	}
	
	@Override
	public double getArea() {
		return getScaledArea(1, 1);
	}
	
	@Override
	public double getLength() {
		return getScaledLength(1, 1);
	}
	
	/**
	 * Default implementation using JTS. Subclasses may replace this with a more efficient implementation.
	 */
	@Override
	public ROI getConvexHull() {
		return GeometryTools.geometryToROI(getGeometry().convexHull(), getImagePlane());
	}
	
	@Override
	public double getSolidity() {
		return isArea() ? getArea() / getConvexHull().getArea() : Double.NaN;
	}
	
//	@Override
//	public double getMaxDiameter() {
//		return getScaledMaxDiameter(1, 1);		
//	}
//	
//	@Override
//	public double getMinDiameter() {
//		return getScaledMinDiameter(1, 1);
//	}
//	
//	@Override
//	public double getScaledMaxDiameter(double pixelWidth, double pixelHeight) {
//		return GeometryROI.computeGeometryStats(getGeometry(), pixelWidth, pixelHeight, false).getMaxDiameter();
//	}
//	
//	@Override
//	public double getScaledMinDiameter(double pixelWidth, double pixelHeight) {
//		return GeometryROI.computeGeometryStats(getGeometry(), pixelWidth, pixelHeight, false).getMinDiameter();
//	}
	
}