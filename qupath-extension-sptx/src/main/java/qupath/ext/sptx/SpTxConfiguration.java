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

package qupath.ext.sptx;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import javafx.beans.property.StringProperty;
import javafx.scene.control.ButtonBar.ButtonData;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.util.Callback;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.gui.QuPathGUI;
// import qupath.lib.gui.commands.interfaces.PathCommand;
import qupath.lib.gui.prefs.PathPrefs;


/**
 * Command used to create and show a suitable dialog box for interactive display of Weka classifiers.
 * 
 * @author Pete Bankhead
 *
 */


public class SpTxConfiguration implements Runnable {
	
	final private static String name = "Create detection classifier (Weka)";
	
	
	final private static StringProperty dianePath = PathPrefs.createPersistentPreference("dianePath", null);
	
	private QuPathGUI qupath;
	

	
	public SpTxConfiguration(final QuPathGUI qupath) {
		this.qupath = qupath;
		// Add Weka path
		// updateExtensionPath();
		// Listen for changes to path property
		// dianePath.addListener((v, o, n) -> updateExtensionPath());
	}
	
//	private void updateExtensionPath() {
//		String path = dianePath.get();
//		if (path != null && new File(path).exists()) {
//			// qupath.addExtensionJar(new File(path));
//		}
//	}
	


    
	@Override
	public void run() {
		// Prompt to select path to Weka
//		if (!Dialogs.showConfirmDialog("Set path to diane.ini", "Do you want to select it manually from your DIAnE installation?"))
//			return;
//		File fileDIAnEIni = QuPathGUI.getDialogHelper(qupath.getStage()).promptForFile("Select diane.ini", null, "AutiPath INI file", new String[]{".ini"});
//		if (fileDIAnEIni == null || !fileDIAnEIni.isFile()) {
//			logger.error("No DIAnE INI file selected.");
//			return;
//		}
		
		final Dialog<Map<String, String>> dialog = new Dialog<>();
		dialog.setTitle("Configuration");
		dialog.setHeaderText("SpTx Analysis");
		dialog.setResizable(true);
		 
		final Label pythonLocationLabel = new Label("Python Location: ");
		final TextField pythonLocationText = new TextField();		
		
		final StringProperty pythonLocationProperty = PathPrefs.createPersistentPreference("pythonLocation", null);

		String pythonLocationString = pythonLocationProperty.get();
		
		pythonLocationText.setText(pythonLocationString);
		         
		GridPane grid = new GridPane();
		grid.add(pythonLocationLabel, 1, 1);
		grid.add(pythonLocationText, 2, 1);		
		dialog.getDialogPane().setContent(grid);
		         
		ButtonType buttonTypeOk = new ButtonType("Ok", ButtonData.OK_DONE);
		dialog.getDialogPane().getButtonTypes().add(buttonTypeOk);
		 
		dialog.setResultConverter((Callback<ButtonType, Map<String, String>>) new Callback<ButtonType, Map<String, String>>() {
		    @Override
		    public Map<String, String> call(ButtonType b) {
		 
		        if (b == buttonTypeOk) {
		        	final Map<String, String> result = new HashMap<String, String>();
		        	result.put("pythonLocation", pythonLocationText.getText());
		        	
		            return result;
		        }
		 
		        return null;
		    }
		});
		         
		Optional<Map<String, String>> result = dialog.showAndWait();
		         
		if (result.isPresent()) {	
			pythonLocationProperty.set(result.get().get("pythonLocation"));
			
		}		
	}
}
