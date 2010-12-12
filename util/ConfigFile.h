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
 * ConfigFile is the object that parses the XML config file and exposes it to
 * the application as an ordered collection of section(s) and element(s).
 */
class ConfigFile {
public:
    /**
     * ConfigFile::Element is the object used for storing a configuration
     * element.
     */
    class Element {
    public:
        Element(std::string & name);
        Element(const char * name);
        Element(ConfigFile::Element & other);
        virtual ~Element();
        std::string & GetName() const;
        bool HasKey(std::string & key);
        bool HasKey(const char * key);
        std::string & GetValue(std::string & key) const;
        std::string & GetValue(const char * key) const;
        std::string & GetValue(std::string & key,
            std::string & default_value) const;
        std::vector<std::string> & GetKeys() const;
        void SetValue(std::string & key, std::string & value);
        void SetValue(const char * key, const char * value);
        void Dump();
        void Dump(int sectionLevel);

        short GetShortValue(const char * key);
        short GetShortValueFromOctal(const char * key);

        bool GetBoolValue(const char * key);
    private:
        std::string *mpName;
        std::map<std::string, std::string> * mpContent;
        std::vector<std::string> * mpContentKeys;
    };

    /**
     * ConfigFile::Section is the object used for storing a configuration
     * section. A Section can have many element(s) and subsection(s).
     */
    class Section {
    public:
        Section(std::string &name);
        Section(const char * name);
        virtual ~Section();

        std::string & GetName() const;

        bool HasElement(std::string & name);
        std::vector<std::string> & GetElements() const;
        void AddElement(Element * element);
        Element & GetElement(std::string & name) const;
        Element & GetElement(const char * name) const;

        bool HasSection(std::string & name);
        std::vector<std::string> & GetSections() const;
        void AddSection(Section * section);
        Section & GetSection(std::string & name) const;
        Section & GetSection(const char * name) const;

        void Dump();
    private:
        Section();
        void Initialize();
        void Dump(int level);
        std::string *mpName;
        std::map<std::string, Element *> *mpElements;
        std::vector<std::string> *mpElementsKeys;
        std::map<std::string, Section *> * mpSubsections;
        std::vector<std::string> * mpSubectionsKeys;
    };


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

