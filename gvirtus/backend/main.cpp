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
 * @file   main.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 12:21:51 2009
 * 
 * @brief  
 * 
 * 
 */

/**
 * @mainpage gVirtuS - A GPGPU transparent virtualization component
 *
 * @section Introduction
 * gVirtuS tries to fill the gap between in-house hosted computing clusters,
 * equipped with GPGPUs devices, and pay-for-use high performance virtual
 * clusters deployed  via public or private computing clouds. gVirtuS allows an
 * instanced virtual machine to access GPGPUs in a transparent way, with an
 * overhead  slightly greater than a real machine/GPGPU setup. gVirtuS is
 * hypervisor independent, and, even though it currently virtualizes nVIDIA CUDA
 * based GPUs, it is not limited to a specific brand technology. The performance
 * of the components of gVirtuS is assessed through a suite of tests in
 * different deployment scenarios, such as providing GPGPU power to cloud
 * computing based HPC clusters and sharing remotely hosted GPGPUs among HPC
 * nodes.
 * 
 * @section License
 * Copyright (C) 2009 - 2010
 *     Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 */

#include <iostream>
#include <algorithm>
#include "ConfigFile.h"
#include "Communicator.h"
#include "Backend.h"

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

#include <stdlib.h>     /* getenv */

using namespace log4cplus;
using namespace std;

vector<string> split(const string& s, const string& f) {
    vector<string> temp;
    if (f.empty()) {
        temp.push_back(s);
        return temp;
    }
    if (s.empty())
        return temp;
    typedef string::const_iterator iter;
    const iter::difference_type f_size(distance(f.begin(), f.end()));
    iter i(s.begin());
    for (iter pos; (pos = search(i, s.end(), f.begin(), f.end())) != s.end();) {
        temp.push_back(string(i, pos));
        advance(pos, f_size);
        i = pos;
    }
    temp.push_back(string(i, s.end()));
    return temp;
}

Logger logger;

int main(int argc, char** argv) {
    //initialize();
    BasicConfigurator config;
    config.configure();
    logger=Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS"));
    LOG4CPLUS_INFO(logger, "GVirtuS backend version" );
    string conf = _CONFIG_FILE;
    if (argc == 2)
        conf = string(argv[1]);
    try {
        LOG4CPLUS_INFO(logger, "Configuration:" << conf.c_str() );
        ConfigFile *cf = new ConfigFile(conf.c_str());
        Communicator *c = Communicator::Get(cf->Get("communicator"));
        vector<string> plugins = split(cf->Get("plugins"), ",");
        Backend b(plugins);
        LOG4CPLUS_INFO(logger, "Up and running" );
        b.Start(c);
        delete c;
        LOG4CPLUS_INFO(logger, "Shutdown" );
    } catch (string &e) {
        LOG4CPLUS_ERROR(logger, "Exception: " << e);
    } catch (const char *e) {
        LOG4CPLUS_ERROR(logger, "Exception: " << e);
    }
    return 0;
}

