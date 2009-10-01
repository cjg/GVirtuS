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
}

ConfigFile::Section::Section(const char * name) {
    mpName = new string(name);
}

ConfigFile::ContentSection::ContentSection(string & name) : Section(name) {
    mpContent = new map<string, string>();
    mpKeys = new vector<string>();
}

ConfigFile::ContentSection::ContentSection(const char * name) : Section(name) {
    mpContent = new map<string, string>();
    mpKeys = new vector<string>();
}

bool ConfigFile::ContentSection::HasKey(string & key) {
    return mpContent->find(key) != mpContent->end();
}

string & ConfigFile::ContentSection::GetValue(std::string & key) const {
    map<string, string>::iterator it;
    it = mpContent->find(key);
    if(it == mpContent->end())
        throw "Key not found!";
    return it->second;
}

string & ConfigFile::ContentSection::GetValue(string & key,
        string & default_value) const {
    map<string, string>::iterator it;
    it = mpContent->find(key);
    if(it == mpContent->end())
        return default_value;
    return it->second;
}

vector<string> & ConfigFile::ContentSection::GetKeys() const {
    return *(mpKeys);
}

void ConfigFile::ContentSection::SetValue(string & key, string & value) {
    if(HasKey(key))
        mpContent->erase(key);
    else
        mpKeys->push_back(key);
    mpContent->insert(make_pair(key, value));
}

ConfigFile::DirSection::DirSection(string & name) : Section(name) {
    mpContent = new map<string, Section *>();
    mpSections = new vector<string>();
}

ConfigFile::DirSection::DirSection(const char * name) : Section(name) {
    mpContent = new map<string, Section *>();
    mpSections = new vector<string>();
}

bool ConfigFile::DirSection::HasSection(string & name) {
    return mpContent->find(name) != mpContent->end();
}

vector<string> & ConfigFile::DirSection::GetSections() const {
    return *(mpSections);
}

void ConfigFile::DirSection::AddSection(ConfigFile::Section * section) {
    if(HasSection(section->GetName()))
        mpContent->erase(section->GetName());
    else
        mpSections->push_back(section->GetName());
    mpContent->insert(make_pair(section->GetName(), section));
}

ConfigFile::Section & ConfigFile::DirSection::GetSection(std::string & name) const {
    map<string, ConfigFile::Section *>::iterator it;
    it = mpContent->find(name);
    if(it == mpContent->end())
        throw "Section not found!";
    return *(it->second);
}

ConfigFile::ConfigFile() {
    mpContent = new ConfigFile::DirSection("top");
}

static int level = 0;
static stack<ConfigFile::Section *> sections;

static void StartElementHandler(void *userData, const XML_Char *name,
        const XML_Char **atts) {
    ConfigFile * pThis = (ConfigFile *) userData;
    cout << "Tag: " << name << endl;
    if(strcasecmp(name, "ConfigFile") == 0) {
        if(!sections.empty()) {
            cerr << "Error: tag 'ConfigFile' is not valid at this point!" << endl;
            return;
        }
        sections.push(pThis->GetTopLevel());
    } else if(strcasecmp(name, "DirSection") == 0) {
        cerr << typeid(sections.top()).name() << endl;
        cerr << typeid(ConfigFile::DirSection&).name() << endl;
        if(sections.empty()
                || typeid(sections.top()) != typeid(ConfigFile::DirSection&)) {
            cerr << "Error: tag 'DirSection' is not valid at this point!" << endl;
            return;
        }
        sections.push(new ConfigFile::DirSection("test"));
    } else if(strcasecmp(name, "ContentSection") == 0) {
        if(sections.empty()
                || typeid(sections.top()) != typeid(ConfigFile::DirSection&)) {
            cerr << "Error: tag 'ContentSection' is not valid at this point!" << endl;
            return;
        }
        sections.push(new ConfigFile::ContentSection("test"));
    } else {
        cerr << "Error: tag '" << name << "' isn't valid!" << endl;
    }
}

static void EndElementHandler(void *userData, const XML_Char *name) {
    ConfigFile * pThis = (ConfigFile *) userData;
    sections.pop();
}

static void CharacterDataHandler(void *userData, const XML_Char *s, int len) {
}


ConfigFile::ConfigFile(const char * name) {
    mpContent = new DirSection("top");
    ifstream in(name);
    string line;

    XML_Parser parser = XML_ParserCreate(NULL);
    XML_SetElementHandler(parser, 
            (XML_StartElementHandler) StartElementHandler,
            (XML_EndElementHandler) EndElementHandler);
    XML_SetCharacterDataHandler(parser,
            (XML_CharacterDataHandler) CharacterDataHandler);
    XML_SetUserData(parser, this);
    
    while(getline(in, line))
        XML_Parse(parser, line.c_str(), line.length(), 0);

    XML_Parse(parser, "", 0, 1);

    XML_ParserFree(parser);

    in.close();
}

