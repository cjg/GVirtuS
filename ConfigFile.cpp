/* 
 * File:   ConfigFile.cpp
 * Author: cjg
 * 
 * Created on October 1, 2009, 12:56 PM
 */

#include <cstring>
#include <fstream>
#include <expat.h>
#include <iostream>
#include <sstream>
#include <stack>
#include "ConfigFile.h"

using namespace std;

/* Element Implementation */
ConfigFile::Element::Element(std::string & name) {
    mpName = new string(name);
    mpContent = new map<string, string > ();
    mpContentKeys = new vector<string > ();
}

ConfigFile::Element::Element(const char* name) {
    mpName = new string(name);
    mpContent = new map<string, string > ();
    mpContentKeys = new vector<string > ();
}

ConfigFile::Element::Element(ConfigFile::Element & other) {
    mpName = new string(*other.mpName);
    mpContent = new map<string, string> (*other.mpContent);
    mpContentKeys = new vector<string> (*other.mpContentKeys);
}


ConfigFile::Element::~Element() {
    delete mpName;
    delete mpContent;
    delete mpContentKeys;
}

std::string & ConfigFile::Element::GetName() const {
    return *(mpName);
}

bool ConfigFile::Element::HasKey(std::string & key) {
    return mpContent->find(key) != mpContent->end();
}

string & ConfigFile::Element::GetValue(std::string & key) const {
    map<string, string>::iterator it;
    it = mpContent->find(key);
    if (it == mpContent->end())
        throw "ConfigFile: Key '" + key + "' not found!";
    return it->second;
}

string & ConfigFile::Element::GetValue(const char * key) const {
    string tmp(key);
    return GetValue(tmp);
}


string & ConfigFile::Element::GetValue(string & key,
        string & default_value) const {
    map<string, string>::iterator it;
    it = mpContent->find(key);
    if (it == mpContent->end())
        return default_value;
    return it->second;
}

vector<string> & ConfigFile::Element::GetKeys() const {
    return *(mpContentKeys);
}

void ConfigFile::Element::SetValue(string & key, string & value) {
    if (HasKey(key))
        mpContent->erase(key);
    else
        mpContentKeys->push_back(key);
    mpContent->insert(make_pair(key, value));
}

void ConfigFile::Element::SetValue(const char * key_, const char * value_) {
    string key(key_), value(value_);
    if (HasKey(key))
        mpContent->erase(key);
    else
        mpContentKeys->push_back(key);
    mpContent->insert(make_pair(key, value));
}

void ConfigFile::Element::Dump() {
    cout << *mpName << " = {" << endl;
    for (map<string, string>::iterator it = mpContent->begin();
            it != mpContent->end(); it++)
        cout << "\t" << it->first << " = '" << it->second << "'" << endl;
    cout << "}" << endl;
}

void ConfigFile::Element::Dump(int sectionLevel) {
    char *spaces = new char[sectionLevel + 1];
    memset(spaces, ' ', sectionLevel);
    spaces[sectionLevel] = 0;
    cout << *spaces << *mpName << " = {" << endl;
    for (map<string, string>::iterator it = mpContent->begin();
            it != mpContent->end(); it++)
        cout << spaces << "\t" << it->first << " = '" << it->second << "'" << endl;
    cout << spaces << "}" << endl;
    delete spaces;
}

short ConfigFile::Element::GetShortValue(const char* key) {
    string value = GetValue(key);
    istringstream iss(value);
    short result;
    if(!(iss >> result))
        throw "ConfigFile: Error converting string to short.";
    return result;
}

/* Section Implementation */
ConfigFile::Section::Section(std::string &name) {
    mpName = new string(name);
    Initialize();
}

ConfigFile::Section::Section(const char * name) {
    mpName = new string(name);
    Initialize();
}

ConfigFile::Section::~Section() {
    delete mpName;
    for (map<string, ConfigFile::Element *>::iterator it = mpElements->begin();
            it != mpElements->end(); it++) 
        delete static_cast<Element *> (it->second);
    delete mpElementsKeys;
    for (map<string, Section *>::iterator it = mpSubsections->begin();
            it != mpSubsections->end(); it++)
        delete static_cast<Section *> (it->second);
    delete mpSubectionsKeys;
}

string & ConfigFile::Section::GetName() const {
    return *(mpName);
}

bool ConfigFile::Section::HasElement(string & name) {
    return mpElements->find(name) != mpElements->end();
}

vector<string> & ConfigFile::Section::GetElements() const {
    return *(mpElementsKeys);
}

void ConfigFile::Section::AddElement(Element * element) {
    if (HasElement(element->GetName()))
        mpElements->erase(element->GetName());
    else
        mpElementsKeys->push_back(element->GetName());
    mpElements->insert(make_pair(element->GetName(), element));
}

ConfigFile::Element & ConfigFile::Section::GetElement(std::string & name) const {
    map<string, ConfigFile::Element *>::iterator it;
    it = mpElements->find(name);
    if (it == mpElements->end())
        throw "Element not found!";
    return *(it->second);
}

ConfigFile::Element & ConfigFile::Section::GetElement(const char * name) const {
    string tmp(name);
    return GetElement(tmp);
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

ConfigFile::Section & ConfigFile::Section::GetSection(string & name) const {
    map<string, ConfigFile::Section *>::iterator it;
    it = mpSubsections->find(name);
    if (it == mpSubsections->end())
        throw "Section not found!";
    return *(it->second);
}

ConfigFile::Section & ConfigFile::Section::GetSection(const char * name) const {
    string tmp(name);
    return GetSection(tmp);
}

void ConfigFile::Section::Dump() {
    Dump(0);
}

void ConfigFile::Section::Initialize() {
    mpSubsections = new map<string, ConfigFile::Section *>();
    mpSubectionsKeys = new vector<string > ();
    mpElements = new map<string, ConfigFile::Element *>();
    mpElementsKeys = new vector<string > ();
}

void ConfigFile::Section::Dump(int level) {
    char *spaces = new char[level + 1];
    memset(spaces, ' ', level);
    spaces[level] = 0;
    cout << spaces << "[" << *mpName << "]" << endl;
    for (map<string, ConfigFile::Element *>::iterator it = mpElements->begin();
            it != mpElements->end(); it++)
        it->second->Dump(level);
    for (map<string, Section *>::iterator it = mpSubsections->begin();
            it != mpSubsections->end(); it++)
        it->second->Dump(level + 1);
}

/* ConfigFile Implementation */
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
        ConfigFile::Element *e = new ConfigFile::Element(tagName);
        for (char **ptr = (char **) atts; *ptr != NULL; ptr += 2)
            e->SetValue(*ptr, *(ptr + 1));
        static_cast<ConfigFile::Section *> (sections.top())->AddElement(e);
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

ConfigFile::~ConfigFile() {
    delete mpContent;
}

void ConfigFile::Dump() {
    mpContent->Dump();
}