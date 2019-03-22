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
 * @file   ConfigFile.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 1 12:56:07 2009
 *
 * @brief
 *
 *
 */

#include "ConfigFile.h"

#include <stdio.h>

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>

using namespace std;

void eatcomments(char *s) {
    char *comments = strchr(s, '#');
    if(comments)
        s[comments - s] = 0;
}

void stripspaces(char *s) {
    unsigned i = 0;
    size_t len = strlen(s);
    for(i = 0; i < len; i++)
        if(!isspace(s[i]))
            break;
    if(i < len) {
        memmove(s, s + i, len - i);
        s[len - i] = 0;
    }
    for(i = strlen(s) - 1; strlen(s) > 0 && i >= 0; i--)
        if(isspace(s[i]))
            s[i] = 0;
        else
            break;
}

bool split(const char *s, char **key, char **value) {
    const char *valueptr = strchr(s, ':');
    if(valueptr == NULL)
        return false;
    *key = (char *) malloc(valueptr - s + 1);
    memmove(*key, s, valueptr - s);
    (*key)[valueptr - s] = 0;
#ifdef _WIN32
    *value = _strdup(valueptr + 1);
#else
    *value = strdup(valueptr + 1);
#endif
    stripspaces(*key);
    stripspaces(*value);
    return true;
}

#ifdef _WIN32
int getline(char **line, size_t *size, FILE *fp) {
	int len = -1;
	char c;
	while(true) {
		if(fread(&c, 1, 1, fp) <= 0)
			return len;
		len++;
		if((unsigned) len >= *size) {
			*size += 256;
			*line = (char *) realloc(*line, *size);
		}
		(*line)[len] = c;
		(*line)[len + 1] = 0;
	}
	return -1;
}
#endif

ConfigFile::ConfigFile(const char* filename) {
#ifndef _WIN32
    FILE *fp = fopen(filename, "r");
#else
	FILE *fp;
	fopen_s(&fp, filename, "r");
#endif
	if(fp == NULL)
		throw "Cannot open gVirtuS config file.";
    char *line = NULL;
    size_t size = 0;
    while(getline(&line, &size, fp) >= 0) {
        eatcomments(line);
        stripspaces(line);
        if(strlen(line) == 0)
            continue;
        char *key, *value;
        if(!split(line, &key, &value))
            throw "Invalid entry in config file.";
        mValues.insert(make_pair(string(key), string(value)));
        free(key);
        free(value);
    }
    free(line);
    fclose(fp);
}

ConfigFile::~ConfigFile() {

}

const string tolower(const std::string &s) {
    string l;
    stringstream ss;

    for(unsigned i = 0; i < s.length(); i++)
        ss << (char)(tolower(s[i]));
    return ss.str();
}

bool ConfigFile::HasKey(const std::string& key) const {
    map<string,string>::const_iterator i = mValues.find(tolower(key));
    return i != mValues.end();
}

const string ConfigFile::Get(const std::string& key) const {
    map<string,string>::const_iterator i = mValues.find(tolower(key));
    if(i == mValues.end())
        throw "Key not found.";
    return i->second;
}

void ConfigFile::Dump() {
    for(map<string,string>::iterator i = mValues.begin(); i != mValues.end();
            i++)
        cout << "ConfigFile[" << i->first << "]: " << i->second << endl;
}
