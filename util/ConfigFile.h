/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   ConfigFile.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 1 12:56:07 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _CONFIGFILE_H
#define	_CONFIGFILE_H

#include <string>
#include <vector>
#include <map>

/**
 * ConfigFile is the object that parses the .properties config file
 * and exposes it to the application as a collection of key(s) and values(s).
 */
class ConfigFile {
public:
    ConfigFile(const char * filename);
    virtual ~ConfigFile();
    bool HasKey(const std::string & key) const;
    const std::string Get(const std::string & key) const;
    void Dump();
private:
    std::map<std::string, std::string> mValues;
};

#endif	/* _CONFIGFILE_H */

