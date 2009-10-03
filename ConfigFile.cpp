/* 
 * File:   ConfigFile.cpp
 * Author: cjg
 * 
 * Created on October 1, 2009, 12:56 PM
 */

#include <fstream>
#include <expat.h>
#include <iostream>
#include <stack>
#include "ConfigFile.h"

using namespace std;

string & ConfigFile::Section::GetName() const {
    return *(mpName);
}

ConfigFile::Section::~Section() {
}

ConfigFile::Section::Section(string & name) {
    mpName = new string(name);
    Initialize();
}

ConfigFile::Section::Section(const char * name) {
    mpName = new string(name);
    Initialize();
}

bool ConfigFile::Section::HasKey(string & key) {
    return mpContent->find(key) != mpContent->end();
}

string & ConfigFile::Section::GetValue(std::string & key) const {
    map<string, string>::iterator it;
    it = mpContent->find(key);
    if (it == mpContent->end())
        throw "Key not found!";
    return it->second;
}

string & ConfigFile::Section::GetValue(string & key,
        string & default_value) const {
    map<string, string>::iterator it;
    it = mpContent->find(key);
    if (it == mpContent->end())
        return default_value;
    return it->second;
}

vector<string> & ConfigFile::Section::GetKeys() const {
    return *(mpContentKeys);
}

void ConfigFile::Section::SetValue(string & key, string & value) {
    if (HasKey(key))
        mpContent->erase(key);
    else
        mpContentKeys->push_back(key);
    mpContent->insert(make_pair(key, value));
}

void ConfigFile::Section::SetValue(const char * key_, const char * value_) {
    string key(key_), value(value_);
    if (HasKey(key))
        mpContent->erase(key);
    else
        mpContentKeys->push_back(key);
    mpContent->insert(make_pair(key, value));
}

bool ConfigFile::Section::HasSection(string & name) {
    return mpSubsections->find(name) != mpSubsections->end();
}

vector<string> & ConfigFile::Section::GetSections() const {
    return *(mpSubectionsKeys);
}

void ConfigFile::Section::AddSection(ConfigFile::Section * section) {
    if (HasSection(section->GetName()))
        mpSubsections->erase(section->GetName());
    else
        mpSubectionsKeys->push_back(section->GetName());
    mpSubsections->insert(make_pair(section->GetName(), section));
}

ConfigFile::Section & ConfigFile::Section::GetSection(std::string & name) const {
    map<string, ConfigFile::Section *>::iterator it;
    it = mpSubsections->find(name);
    if (it == mpSubsections->end())
        throw "Section not found!";
    return *(it->second);
}

void ConfigFile::Section::Dump() {
    Dump(0);
}

void ConfigFile::Section::Initialize() {
    mpContent = new map<string, string > ();
    mpContentKeys = new vector<string > ();
    mpSubsections = new map<string, ConfigFile::Section *>();
    mpSubectionsKeys = new vector<string > ();
}

void ConfigFile::Section::Dump(int level) {
    char *spaces = new char[level + 1];
    memset(spaces, ' ', level);
    spaces[level] = 0;
    cout << spaces << "[" << *mpName << "]" << endl;
    for (map<string, string>::iterator it = mpContent->begin();
            it != mpContent->end(); it++)
        cout << spaces << it->first << " = '" << it->second << "'" << endl;
    for (map<string, Section *>::iterator it = mpSubsections->begin();
            it != mpSubsections->end(); it++)
        it->second->Dump(level + 1);
}

ConfigFile::ConfigFile() {
    mpContent = new ConfigFile::Section("config");
}

static stack<ConfigFile::Section *> sections;

static void StartElementHandler(void *userData, const XML_Char *name,
        const XML_Char **atts) {
    ConfigFile * pThis = (ConfigFile *) userData;
    char *tagName = NULL;
    for (char **ptr = (char **) atts; *ptr != NULL; ptr += 2)
        if (strcasecmp(*ptr, "name") == 0)
            tagName = *(ptr + 1);
    cout << "Tag: " << name << endl;
    if (strcasecmp(name, "config") == 0) {
        if (!sections.empty()) {
            cerr << "Error: tag '" << name << "' not valid at this point!"
                    << endl;
            return;
        }
        sections.push(pThis->GetTopLevel());
    } else if (strcasecmp(name, "section") == 0) {
        if (sections.empty()) {
            cerr << "Error: tag '" << name << "' not valid at this point!"
                    << endl;
            return;
        }
        if (tagName == NULL) {
            cerr << "Error: the tag '" << name << "' must have a name!" << endl;
            return;
        }
        ConfigFile::Section *s = new ConfigFile::Section(tagName);
        ((ConfigFile::Section *) sections.top())->AddSection(s);
        sections.push(s);
    } else if (strcasecmp(name, "element") == 0) {
        if (sections.empty()) {
            cerr << "Error: tag '" << name << "' not valid at this point!"
                    << endl;
            return;
        }
        if (tagName == NULL) {
            cerr << "Error: the tag '" << name << "' must have a name!" << endl;
            return;
        }
        for (char **ptr = (char **) atts; *ptr != NULL; ptr += 2)
            ((ConfigFile::Section *) sections.top())->SetValue(*ptr, *(ptr + 1));
    } else {
        cerr << "Error: tag '" << name << "' not valid!" << endl;
    }
}

static void EndElementHandler(void *userData, const XML_Char *name) {
    if (strcasecmp(name, "config") == 0 || strcasecmp(name, "section") == 0)
        sections.pop();
}

static void CharacterDataHandler(void *userData, const XML_Char *s, int len) {
}

ConfigFile::ConfigFile(const char * name) {
    mpContent = new Section("config");
    ifstream in(name);
    string line;

    XML_Parser parser = XML_ParserCreate(NULL);
    XML_SetElementHandler(parser,
            (XML_StartElementHandler) StartElementHandler,
            (XML_EndElementHandler) EndElementHandler);
    XML_SetCharacterDataHandler(parser,
            (XML_CharacterDataHandler) CharacterDataHandler);
    XML_SetUserData(parser, this);

    while (getline(in, line))
        XML_Parse(parser, line.c_str(), line.length(), 0);

    XML_Parse(parser, "", 0, 1);

    XML_ParserFree(parser);

    in.close();
}

