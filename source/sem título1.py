#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:44:38 2019

@author: hmonteiro
"""

        # correct for Av bias
        pars[2,:] = pars[2,:] - (-0.12646707*pars[3,:] +  0.0492754)
