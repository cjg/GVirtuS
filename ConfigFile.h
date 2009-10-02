/* 
 * File:   ConfigFile.h
 * Author: cjg
 *
 * Created on October 1, 2009, 12:56 PM
 */

#ifndef _CONFIGFILE_H
#define	_CONFIGFILE_H

#include <string>
#include <vector>
#include <map>

class ConfigFile {
public:
    class Section {
    public:
        Section(std::string &name);
        Section(const char * name);
        virtual ~Section();
        std::string & GetName() const;
        bool HasKey(std::string & key);
        std::string & GetValue(std::string & key) const;
        std::string & GetValue(std::string & key,
            std::string & default_value) const;
        std::vector<std::string> & GetKeys() const;
        void SetValue(std::string & key, std::string & value);
        void SetValue(const char * key, const char * value);
        bool HasSection(std::string & name);
        std::vector<std::string> & GetSections() const;
        void AddSection(Section * section);
        Section & GetSection(std::string & name) const;
        void Dump();
    private:
        Section();
        void Initialize();
        void Dump(int level);
        std::string *mpName;
        std::map<std::string, std::string> *mpContent;
        std::vector<std::string> *mpContentKeys;
        std::map<std::string, Section *> * mpSubsections;
        std::vector<std::string> * mpSubectionsKeys;
    };

public:
    ConfigFile();
    ConfigFile(const char * filename);
    Section * GetTopLevel() {
        return mpContent;
    }
private:
    Section *mpContent;
};

#endif	/* _CONFIGFILE_H */

