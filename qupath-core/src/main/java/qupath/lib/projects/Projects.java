/*-
 * #%L
 * This file is part of QuPath.
 * %%
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

package qupath.lib.projects;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.util.Comparator;

/**
 * Static methods to help when working with Projects.
 * 
 * @author Pete Bankhead
 *
 */
public class Projects {

	static class ImageEntryComparator implements Comparator<ProjectImageEntry<?>> {
	
		static ImageEntryComparator instance = new ImageEntryComparator();
		
		@Override
		public int compare(ProjectImageEntry<?> o1, ProjectImageEntry<?> o2) {
			String n1 = o1.getImageName();
			String n2 = o2.getImageName();
			if (n1 == null) {
				if (n2 == null)
					return 0;
				else
					return 1;
			} else if (n2 == null)
				return -1;
			return n1.compareTo(n2);
		}
		
	}

	/**
	 * Get the base directory of a project stored on the local file system.
	 * 
	 * @param project
	 * @return the base directory, or null if no directory could be found.
	 */
	public static File getBaseDirectory(Project<?> project) {
		Path path = project == null ? null : project.getPath();
		if (path == null)
			return null;
		File file = path.toFile();
		if (file.isDirectory())
			return file;
		file = file.getParentFile();
		return file.isDirectory() ? file : null;
	}

	/**
	 * Create a new project using the local file system. It is assumed that the directory of the project file 
	 * can be managed by the project, and therefore it should not contain additional user files.
	 * 
	 * @param file either the project file, or a directory that should contain a project file with a default name
	 * @param cls generic type for the project (only BufferedImage currently supported)
	 * @return
	 */
	public static <T> Project<T> createProject(File file, Class<T> cls) {
		if (file.isDirectory())
			file = new File(file, ProjectIO.DEFAULT_PROJECT_NAME + ProjectIO.getProjectExtension(true));
		if (cls == BufferedImage.class)
			return (Project<T>)new DefaultProject(file);
		throw new IllegalArgumentException("Cannot create project for " + cls);
	}

}