//  KhuCvApp.h: interface of CKhuCvApp
//	Dept. Software Convergence, Kyung Hee University
//	Prof. Daeho Lee, nize@khu.ac.kr
//

#pragma once
#include <wx/wx.h>
#include <wx/aui/aui.h>
#include <wx/clipbrd.h>
#include <wx/listctrl.h>
#include <wx/dir.h>
#include <wx/filehistory.h>
#include <wx/config.h>
#include <wx/filename.h>
#include <wx/stdpaths.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/core/utility.hpp>
#include "opencv2/video.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


#include <onnxruntime_cxx_api.h>

#include <complex>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <fstream>

#include "MainFrame.h"
#include "ClientView.h"
#include "ChildFrame.h"
#include "MainDialog.h"

class CKhuCvApp : public wxApp
{
public:
    CMainFrame* m_pMainFrame;
    CMainDialog* m_pMainDialog;
    wxLocale m_locale;

    ~CKhuCvApp();
    virtual bool OnInit();
};

void NewFileOpen(const char* fileName, cv::Mat cvImage, int nPosX = 0, int nPosY = 0);
void DisplayImage(cv::Mat cvImage, int nPosX, int nPosY, bool bErase, bool bClearPos);
CKcImage GetLastSelImage();
CMainDialog* GetMainDialog();
void DlgPrintf(const char* ptr, ...);
