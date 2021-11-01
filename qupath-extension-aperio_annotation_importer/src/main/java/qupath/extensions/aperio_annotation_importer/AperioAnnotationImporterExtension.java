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

import javafx.scene.control.Menu;
import qupath.lib.gui.ActionTools;
import qupath.lib.gui.QuPathGUI;
// import qupath.lib.gui.commands.OpenWebpageCommand;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.tools.MenuTools;

/**
 * QuPath extension for creating detection classifiers using Weka.
 * 
 * @author Pete Bankhead
 *
 */
public class AperioAnnotationImporterExtension implements QuPathExtension {
	         
	@Override
	public void installExtension(QuPathGUI qupath) {
		Menu menu = qupath.getMenu("File>Import Annotations...", true);
//		MenuTools.addMenuItems(
//				menu,
//				ActionTools.createAction(new OpenWebpageCommand(qupath, "http://www.cs.waikato.ac.nz/ml/weka/downloading.html"), "Download Weka (web)")				
//				);
		
		// Menu menuClassify = qupath.getMenu("Classify", true);
		MenuTools.addMenuItems(
				menu,
				ActionTools.createAction(new AperioAnnotationImporterCommand(qupath), "Aperio Annotation (XML) Importer")
				);
	}

	@Override
	public String getName() {
		return "Aperio annotation (XML) importer extension";
	}

	@Override
	public String getDescription() {
		return "Adds support for Aperio annotation";
	}
	
}