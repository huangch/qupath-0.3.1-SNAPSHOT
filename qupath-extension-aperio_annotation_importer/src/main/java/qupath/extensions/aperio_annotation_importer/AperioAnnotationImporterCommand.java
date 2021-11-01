/*-
 * #%L
 * This file is part of a QuPath extension.
 * %%
 * Copyright (C) 2014 - 2016 The Queen's University of Belfast, Northern Ireland
 * Contact: IP Management (ipmanagement@qub.ac.uk)
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 * #L%
 */

package qupath.extensions.aperio_annotation_importer;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.xml.sax.SAXException;

import javafx.beans.property.StringProperty;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.control.ButtonBar.ButtonData;
import javafx.scene.layout.GridPane;
import javafx.stage.Stage;
import javafx.util.Callback;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.gui.scripting.QPEx;
import qupath.lib.images.ImageData;
import qupath.lib.measurements.MeasurementList;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.geom.Point2;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Command used to create and show a suitable dialog box for interactive display of Weka classifiers.
 * 
 * @author Pete Bankhead
 *
 */
@SuppressWarnings("deprecation")
public class AperioAnnotationImporterCommand implements Runnable {
	final private static String name = "Aperio Annotation (XML) Importer";

	final private static Logger logger = LoggerFactory.getLogger(AperioAnnotationImporterCommand.class);

	// final private static StringProperty wekaPath = PathPrefs.createPersistentPreference("wekaPath", null);
	private File xmlFileName;
	
	
	final private QuPathGUI qupath;
	
	private Stage dialog;
	
	public AperioAnnotationImporterCommand(final QuPathGUI qupath) {
		this.qupath = qupath;
	}

	@Override
	public void run() {
        // Load the input XML document, parse it and return an instance of the
        // Document class.
        try {
    		dialog = new Stage();
    		if (qupath != null)
    			dialog.initOwner(qupath.getStage());
    		dialog.setTitle(name);
    		
    		xmlFileName = Dialogs.getChooser(dialog).promptForFile("Select Aperio Annotation (XML) file", null, "Aperio Annotation (XML) file", new String[]{".xml"});
    		
    		
    		
    		
    		
    		
    		
    		
//    		final Dialog<Map<String, String>> dialog = new Dialog<>();
//    		dialog.setTitle("Configuration");
//    		dialog.setHeaderText("AutopathServer");
//    		dialog.setResizable(true);
//    		 
//    		final Label xShiftLabel = new Label("X-shift: ");
//    		final Label yShiftLabel = new Label("Y-shift: ");
//    		
//    		final TextField xShiftText = new TextField();
//    		final TextField yShiftText = new TextField();
//    		         
//    		GridPane grid = new GridPane();
//    		grid.add(xShiftLabel, 1, 1);
//    		grid.add(xShiftText, 2, 1);
//    		grid.add(yShiftLabel, 1, 2);
//    		grid.add(yShiftText, 2, 2);
//    		dialog.getDialogPane().setContent(grid);
//    		         
//    		ButtonType buttonTypeOk = new ButtonType("Ok", ButtonData.OK_DONE);
//    		dialog.getDialogPane().getButtonTypes().add(buttonTypeOk);
//    		 
//    		dialog.setResultConverter((Callback<ButtonType, Map<String, String>>) new Callback<ButtonType, Map<String, String>>() {
//    		    @Override
//    		    public Map<String, String> call(ButtonType b) {
//    		 
//    		        if (b == buttonTypeOk) {
//    		        	final Map<String, String> result = new HashMap<String, String>();
//    		        	result.put("xshift", xShiftText.getText());
//    		        	result.put("yshift", yShiftText.getText());
//    		        	
//    		            return result;
//    		        }
//    		 
//    		        return null;
//    		    }
//    		});
//    		         
//    		Optional<Map<String, String>> result = dialog.showAndWait();
    		         
    		double xshift = 0;
    		double yshift = 0;
    				
//    		if (result.isPresent()) {
//    			xshift = Double.parseDouble(result.get().get("xshift"));
//    			yshift = Double.parseDouble(result.get().get("yshift"));
//    		}
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		final ImageData<BufferedImage> imageData = (ImageData<BufferedImage>)QPEx.getCurrentImageData();
    		final PathObjectHierarchy hierarchy = imageData.getHierarchy();
    		final List<PathObject> pathObjList = new ArrayList();
            final DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            final DocumentBuilder builder = factory.newDocumentBuilder();
            final Document document = builder.parse(xmlFileName);
            
            final NodeList annotationsContentList = document.getDocumentElement().getChildNodes();
            
            for (int i = 0; i < annotationsContentList.getLength(); i++) {
                final Node annotationsContent = annotationsContentList.item(i);
            	if (annotationsContent.getNodeType() != Node.ELEMENT_NODE) continue;
  
                if (annotationsContent.getNodeName().equals("Annotation")) {
                	final Node annotation = annotationsContent;
				    final NamedNodeMap annotationAttrs = annotation.getAttributes();
				    final String annotationId = annotationAttrs.getNamedItem("Id").getNodeValue();
				    final String annotationName = annotationAttrs.getNamedItem("Name").getNodeValue();
				    
				    final HashMap<String, String> annotationMeasurementMap = new HashMap<>(); 
                	
                	final NodeList annotationContentList = annotation.getChildNodes();
                	
                	for(int j = 0; j < annotationContentList.getLength(); j++) {
                		final Node annotationContent = annotationContentList.item(j);
                		if (annotationContent.getNodeType() != Node.ELEMENT_NODE) continue;
                		
            			if (annotationContent.getNodeName().equals("Attributes")) {
            				final Node attributes = annotationContent;
            				
            				final NodeList attributesContentList = attributes.getChildNodes();
            				
            				for(int k = 0; k < attributesContentList.getLength(); k++) {
            					final Node attributesContent = attributesContentList.item(k);
            					if (attributesContent.getNodeType() != Node.ELEMENT_NODE) continue;
            					
            					if (attributesContent.getNodeName().equals("Attribute")) {
            						final Node attribute = attributesContent;
                					final NamedNodeMap attributeAttrs = attribute.getAttributes();
                					final String attributeName = attributeAttrs.getNamedItem("Name").getNodeValue();
                					final String attributeValue = attributeAttrs.getNamedItem("Value").getNodeValue();
            						
            						annotationMeasurementMap.put(attributeName, attributeValue);
            					}
            				}
            			}
                		else if (annotationContent.getNodeName().equals("Regions")) {
                			final Node regions = annotationContent;
                			
                			final NodeList regionsContentList = regions.getChildNodes();
                			
                			for(int k = 0; k < regionsContentList.getLength(); k++) {
                				final Node regionsContent = regionsContentList.item(k);
                				if (regionsContent.getNodeType() != Node.ELEMENT_NODE) continue;
                				
                				if (regionsContent.getNodeName().equals("Region")) {
                					final Node region = regionsContent;
                					
                					final NamedNodeMap regionAttrs = region.getAttributes();
                					final String regionType = regionAttrs.getNamedItem("Type").getNodeValue();
                					final String regionId = regionAttrs.getNamedItem("Id").getNodeValue();
                					
                					final NodeList regionContentList = region.getChildNodes();
                					
                					for(int l = 0; l < regionContentList.getLength(); l++) {
                						final Node regionContent = regionContentList.item(l);
                						if (regionContent.getNodeType() != Node.ELEMENT_NODE) continue;
                						
                						if (regionContent.getNodeName().equals("Vertices")) {
                							final Node vertices = regionContent;
                							
                							final NodeList verticesContentList = vertices.getChildNodes();
                							
                							final List<Point2> points = new ArrayList();
                							
                							for(int m = 0; m < verticesContentList.getLength(); m++) {
                								final Node verticesContent = verticesContentList.item(m);
                								if (verticesContent.getNodeType() != Node.ELEMENT_NODE) continue;
                								
                								final Node vertex = verticesContent;
                								
                							    final NamedNodeMap vertexAttrs = vertex.getAttributes();
                							    final String X = vertexAttrs.getNamedItem("X").getNodeValue();
                							    final String Y = vertexAttrs.getNamedItem("Y").getNodeValue();
                							    points.add(new Point2(Double.parseDouble(X)+xshift, Double.parseDouble(Y)+yshift));
                							}
                							
                							final String pathObjName = (annotationName.equals("")? annotationId: annotationId+" ("+annotationName+")")+":"+regionId;
                							final String pathClsName = "aperio";
                									
                							ROI pathRoi = null;
                							if(regionType.equals("0")) pathRoi = ROIs.createPolygonROI(points, null);
                							else pathRoi = ROIs.createPolylineROI(points, null);
                							
                					    	final PathClass pathCls = PathClassFactory.getPathClass(pathClsName);
                					    	
                					    	
                					    	final PathAnnotationObject pathObj = (PathAnnotationObject) PathObjects.createAnnotationObject(pathRoi, pathCls);
                					    	// final PathObject pathObj = pathObj.createAnnotationObject(pathRoi, pathCls);
                					    	pathObj.setName(pathObjName);
                					    	
                							final MeasurementList pathObjMeasList = pathObj.getMeasurementList();

                							annotationMeasurementMap.forEach((annotMeasName, annotMeasValue) -> {
                								if(!annotMeasName.isBlank()) {
                									final String attrName = annotMeasValue.isBlank()? annotMeasName: annotMeasName+"="+annotMeasValue;
                									pathObjMeasList.addMeasurement(attrName, 1);
                									// 	pathObj.storeAttribute("aperio_key:"+annotMeasName, "aperio_value"+annotMeasValue);
                								}
                					        });
                							
                							pathObjMeasList.close();

                					    	pathObjList.add(pathObj);   
                						}
                					}
            					}
                			}
                		}
                	}
                }
            }
            
	        hierarchy.addPathObjects(pathObjList);
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException | ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	

	

	
}
