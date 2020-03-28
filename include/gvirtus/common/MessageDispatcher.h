/*
 * File:   MessageDispatcher.h
 * Author: cpalmieri
 *
 * Created on March 4, 2016, 2:22 PM
 */

#pragma once

namespace gvirtus::common {
class MessageDispatcher {
 public:
  MessageDispatcher();
  MessageDispatcher(const MessageDispatcher &orig);
  virtual ~MessageDispatcher();

 private:
};
}
