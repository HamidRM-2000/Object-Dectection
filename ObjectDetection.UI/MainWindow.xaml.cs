using Microsoft.Win32;
using ObjectDetection.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ObjectDetection.UI
{
   /// <summary>
   /// Interaction logic for MainWindow.xaml
   /// </summary>
   public partial class MainWindow : Window
   {
      public MainWindow()
      {
         InitializeComponent();
      }

      private async void Button_Click(object sender, RoutedEventArgs e)
      {
         FileDialog dialog = new OpenFileDialog();
         dialog.Filter = "(*.jpg , *.jpeg , *.png ) | *.jpg; *.jpeg; *.png";

         if ((bool)dialog.ShowDialog())
         {
            txtAlert.Text = "Image In Proccess ...";
            ModelInput input = new ModelInput() { ImagePath = dialog.FileName };

            var result = await ConsumeModel.PredictAsync(input, true);

            ImageControl.Source = new BitmapImage(new Uri(input.PredictedImagePath));
            txtAlert.Text = $"Found {result.Objects} Objects in {result.Labels.Length} Categories";
         }
      }
   }
}
