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
        std::string & GetName() const;
        virtual ~Section();
    protected:
        Section(std::string &name);
        Section(const char * name);
    private:
        Section();
        std::string *mpName;
    };

    class ContentSection : public Section {
    public:
        ContentSection(std::string &name);
        ContentSection(const char * name);
        bool HasKey(std::string & key);
        std::string & GetValue(std::string & key) const;
        std::string & GetValue(std::string & key, 
            std::string & default_value) const;
        std::vector<std::string> & GetKeys() const;
        void SetValue(std::string & key, std::string & value);
    private:
        std::map<std::string, std::string> *mpContent;
        std::vector<std::string> *mpKeys;
    };

    class DirSection : public Section {
    public:
        DirSection(std::string &name);
        DirSection(const char *name);
        bool HasSection(std::string & name);
        std::vector<std::string> & GetSections() const;
        void AddSection(Section * section);
        Section & GetSection(std::string & name) const;
    private:
        std::map<std::string, Section *> * mpContent;
        std::vector<std::string> * mpSections;
    };

public:
    ConfigFile();
    ConfigFile(const char * filename);
    Section * GetTopLevel() {
        return mpContent;
    }
private:
    DirSection *mpContent;
};

#endif	/* _CONFIGFILE_H */

