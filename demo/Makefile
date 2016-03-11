# Environment 

SUBDIRS = matMulMPICUDA matMulTHRCUDA


subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ 

clean:
	for dir in $(SUBDIRS); do \
          $(MAKE) -C $$dir clean; \
	done 

.PHONY: clean subdirs $(SUBDIRS)
